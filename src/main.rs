#![feature(let_chains)]

use std::arch::x86_64::{
    __m128, _mm_and_ps, _mm_broadcast_ss, _mm_castsi128_ps, _mm_cmpneq_ps, _mm_div_ps,
    _mm_loadu_ps, _mm_mul_ps, _mm_set1_epi32, _mm_sub_ps,
};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroU32;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use nalgebra::Vector3;

#[derive(Clone, Copy, Debug)]
enum Node {
    Split { interm_idx: u32 },
    Final { particle_idx: u32 },
}
impl Node {
    fn from_raw_index(i: RawIndex) -> Option<Self> {
        if i == 0 {
            None
        } else if i > 0 {
            Some(Self::Final {
                particle_idx: i as u32,
            })
        } else {
            Some(Self::Split {
                interm_idx: (-i) as u32,
            })
        }
    }
    fn raw(self) -> RawIndex {
        match self {
            Self::Final { particle_idx } => particle_idx as i32,
            Self::Split { interm_idx } => -(interm_idx as i32),
        }
    }
}
#[derive(Debug)]
struct Tree {
    root: RawIndex,
    root_level: i8,
    deepest: i8,
    mid: Vectorf,
    mw: Coord,
    len: u32,
}
type RawIndex = i32;
type Coord = f32;
const NULL_IDX: RawIndex = 0;

const RADIUS: Coord = 0.05;

fn vector_to_idx(vector: &Vectorf) -> u8 {
    u8::from(vector.x >= 0.0) | (u8::from(vector.y >= 0.0) << 1) | (u8::from(vector.z >= 0.0) << 2)
}
fn idx_to_vector(idx: u8) -> Vectorf {
    Vectorf::new(
        if idx & 1 == 1 { 1.0 } else { -1.0 },
        if idx & 2 == 2 { 1.0 } else { -1.0 },
        if idx & 4 == 4 { 1.0 } else { -1.0 },
    )
}

impl Tree {
    fn validate(
        node: RawIndex,
        middle: Vectorf,
        next_gran: Coord,
        all: &mut Vec<u32>,
        particles: &[Vectorf],
        arena: &[[RawIndex; 8]],
        mids: &[Vectorf],
    ) {
        match Node::from_raw_index(node) {
            None => (),
            Some(Node::Split { interm_idx }) => {
                let mut sub = [const { Vec::new() }; 8];
                for i in 0..8 {
                    Self::validate(
                        arena[interm_idx as usize][i],
                        middle + idx_to_vector(i as u8) * next_gran,
                        next_gran * 0.5,
                        &mut sub[i],
                        particles,
                        arena,
                        mids,
                    );
                }
                for i in 0..8 {
                    for p in &sub[i] {
                        let p = *p as usize;
                        let xbounds = if i & 1 == 1 {
                            middle.x..middle.x + next_gran * 2.0
                        } else {
                            middle.x - next_gran * 2.0..middle.x
                        };
                        let ybounds = if i & 2 == 2 {
                            middle.y..middle.y + next_gran * 2.0
                        } else {
                            middle.y - next_gran * 2.0..middle.y
                        };
                        let zbounds = if i & 4 == 4 {
                            middle.z..middle.z + next_gran * 2.0
                        } else {
                            middle.z - next_gran * 2.0..middle.z
                        };
                        assert!(xbounds.contains(&particles[p].x));
                        assert!(ybounds.contains(&particles[p].y));
                        assert!(zbounds.contains(&particles[p].z));
                    }
                }
                for mut s in sub {
                    all.append(&mut s);
                }
            }
            Some(Node::Final { particle_idx }) => {
                all.push(particle_idx);
            }
        };
    }
    pub fn insert(
        &mut self,
        index: NonZeroU32,
        all_positions: &[Vectorf],
        arena: &mut Vec<[RawIndex; 8]>,
        mids: &mut Vec<Vectorf>,
    ) {
        let index = index.get();
        assert!(index < i32::MAX as u32);

        if self.root == NULL_IDX {
            self.root = Node::Final {
                particle_idx: index,
            }
            .raw();
            self.len += 1;
            return;
        }
        let root_shift = all_positions[index as usize] - self.mid;

        let mut level = self.root_level;

        if root_shift.abs().max() > Coord::powi(2.0, self.root_level.into()) {
            unreachable!();
        }

        enum CurrentNode {
            Root,
            Split { interm: u32, child: u8 },
        }
        let mut current_node = CurrentNode::Root;
        let mut width = Coord::powi(2.0, self.root_level.into());
        let mut mid = self.mid;

        loop {
            //dbg!(&self, mid, index, level);
            let n = match current_node {
                CurrentNode::Root => self.root,
                CurrentNode::Split { interm, child } => arena[interm as usize][child as usize],
            };
            let (children, interm) = match Node::from_raw_index(n).unwrap() {
                Node::Final { particle_idx } => {
                    if particle_idx == index {
                        unreachable!("already inserted");
                    } else {
                        let split_idx = arena.len() as u32;
                        match current_node {
                            CurrentNode::Root => {
                                self.root = Node::Split {
                                    interm_idx: split_idx,
                                }
                                .raw()
                            }
                            CurrentNode::Split { interm, child } => {
                                arena[interm as usize][child as usize] = Node::Split {
                                    interm_idx: split_idx,
                                }
                                .raw()
                            }
                        }

                        arena.push([NULL_IDX; 8]);
                        mids.push(mid);
                        let old_idx = vector_to_idx(&(all_positions[particle_idx as usize] - mid));
                        arena[split_idx as usize][usize::from(old_idx)] =
                            Node::Final { particle_idx }.raw();

                        //dbg!(old_idx);
                        (&mut arena[split_idx as usize], split_idx)
                    }
                }
                Node::Split { interm_idx } => (&mut arena[interm_idx as usize], interm_idx),
            };

            let aabb_size = width * 0.5;
            let child_idx = usize::from(vector_to_idx(&(all_positions[index as usize] - mid)));
            //dbg!(child_idx);
            let child = &mut children[child_idx];

            if *child == NULL_IDX {
                *child = Node::Final {
                    particle_idx: index,
                }
                .raw();
                self.len += 1;
                return;
            }

            let direction = Vectorf::new(
                if child_idx & 1 == 1 { 1.0 } else { -1.0 },
                if child_idx & 2 == 2 { 1.0 } else { -1.0 },
                if child_idx & 4 == 4 { 1.0 } else { -1.0 },
            );
            level -= 1;
            self.deepest = self.deepest.min(level);
            self.mw = self.mw.min(aabb_size);
            mid += direction * aabb_size;
            width = aabb_size;
            current_node = CurrentNode::Split {
                interm,
                child: child_idx as u8,
            };
        }

        //self.root.as_mut().unwrap().insert(&self.mid, index, , all_positions);
    }
    pub fn neighbor_radius_search(
        &self,
        particles: &[Vectorf],
        arena: &[[RawIndex; 8]],
        mids: &[Vectorf],
    ) -> Vec<(u32, u32)> {
        // TODO: parallelize

        #[derive(Debug)]
        enum Type {
            MultiMulti,
            SingleMulti,
        }
        struct Work {
            work: Vec<[u32; 2]>,
            ty: Type,
            width: Coord,
        }

        struct State {
            work: Mutex<Vec<Work>>,
            finished: Mutex<Vec<Vec<(u32, u32)>>>,
            semaphore: AtomicUsize,
        }

        let initial_work = match Node::from_raw_index(self.root) {
            None => return Vec::new(),
            Some(Node::Final { particle_idx }) => return vec![(particle_idx, particle_idx)],
            Some(Node::Split { interm_idx }) => Work {
                work: vec![[interm_idx, interm_idx]],
                width: Coord::powi(2.0, self.root_level.into()),
                ty: Type::MultiMulti,
            },
        };
        let state = State {
            work: Mutex::new(vec![initial_work]),
            finished: Mutex::new(Vec::new()),
            semaphore: AtomicUsize::new(NUM_THREADS),
        };

        let abs_mask: __m128 = unsafe { _mm_castsi128_ps(_mm_set1_epi32(!((1 << 31) as i32))) };

        let do_mm_drain = |mm: Vec<[u32; 2]>, width: Coord, state: &State| {
            let vec2w = unsafe { _mm_broadcast_ss(&(2.0 * width)) };

            let mut next_sm = Vec::with_capacity(WORK_SIZE);
            let mut next_mm = Vec::with_capacity(WORK_SIZE);
            let mut next_fin = Vec::with_capacity(WORK_SIZE);

            for [i1, i2] in mm {
                // Append Cartesian product of respective subnodes, filtering out the ones
                // that cannot be within range at all. We don't necessarily need to
                // calculate the exact minimum distance between the cubes, as long as a
                // valid approximation is used.

                // This part is very hot, so I used SIMD directly, improving perf by ~3%

                if CRAZY_SIMD {
                    unsafe {
                        // XXX: Very unsafe, since Vector3 is 12-byte, but works :) 4th coordinate is
                        // dont-care in this case

                        // Load mid-points of respective (equally sized) boxes.
                        let mid1 = _mm_loadu_ps(mids.as_ptr().add(i1 as usize).cast());
                        let mid2 = _mm_loadu_ps(mids.as_ptr().add(i2 as usize).cast());

                        let diff = _mm_sub_ps(mid2, mid1); // distance between
                        let scale = _mm_and_ps(abs_mask, diff); // abs(diff)

                        // mask: effs if and only if mid2 != mid1, otherwise zeros
                        // for floating point reasons, mid2 and mid1 must be compared directly rather
                        // than comparing their difference diff
                        let eqm = _mm_cmpneq_ps(mid2, mid1);
                        // divide, filtering away division by zero (i.e. the two boxes have the same
                        // coordinate wrt any axis).
                        let diag = _mm_and_ps(eqm, _mm_div_ps(diff, scale));

                        // calculate diff + diag + 2 * width
                        let fin = _mm_sub_ps(diff, _mm_mul_ps(diag, vec2w));
                        // square before summing to get norm
                        let finsq = _mm_mul_ps(fin, fin);

                        let [x, y, z, _w]: [f32; 4] = std::mem::transmute(finsq);
                        // check if outside
                        // inclusive inequality is allowed for rejection, since the octree boxes are
                        // half-open intervals, so the extreme case is treated as 'outside'
                        if x + y + z >= 1.0 {
                            continue;
                        }
                    }
                } else {
                    let mid1 = mids[i1 as usize];
                    let mid2 = mids[i2 as usize];

                    let diff = mid2 - mid1;
                    let scale = diff.abs();

                    // TODO: more efficient method
                    let diag = Vectorf::new(
                        if mid1.x == mid2.x {
                            0.0
                        } else {
                            diff.x / scale.x
                        },
                        if mid1.y == mid2.y {
                            0.0
                        } else {
                            diff.y / scale.y
                        },
                        if mid1.z == mid2.z {
                            0.0
                        } else {
                            diff.z / scale.z
                        },
                    );
                    if (diff - diag * width * 2.0).norm_squared() >= 1.0 {
                        continue;
                    }
                }

                for i in 0..8 {
                    let j_range = if i1 == i2 {
                        // node onto itself
                        i..8
                    } else {
                        // independent nodes
                        0..8
                    };

                    let Some(k1) = Node::from_raw_index(arena[i1 as usize][i as usize]) else {
                        continue;
                    };
                    for j in j_range {
                        let Some(k2) = Node::from_raw_index(arena[i2 as usize][j as usize]) else {
                            continue;
                        };
                        match (k1, k2) {
                            (
                                Node::Final { particle_idx: f1 },
                                Node::Final { particle_idx: f2 },
                            ) => {
                                if (particles[f1 as usize] - particles[f2 as usize]).norm_squared()
                                    <= 1.0
                                {
                                    next_fin.push((f1, f2));
                                }
                            }
                            (Node::Split { interm_idx: i1 }, Node::Split { interm_idx: i2 }) => {
                                next_mm.push([i1, i2])
                            }
                            (Node::Final { particle_idx: p1 }, Node::Split { interm_idx: i2 })
                            | (Node::Split { interm_idx: i2 }, Node::Final { particle_idx: p1 }) => {
                                next_sm.push([p1, i2])
                            }
                        }
                    }
                }
                if next_fin.len() >= WORK_SIZE {
                    let chunk = std::mem::replace(&mut next_fin, Vec::with_capacity(WORK_SIZE));
                    state.finished.lock().unwrap().push(chunk);
                }
                if next_sm.len() >= WORK_SIZE {
                    let chunk = std::mem::replace(&mut next_sm, Vec::with_capacity(WORK_SIZE));
                    state.work.lock().unwrap().push(Work {
                        work: chunk,
                        ty: Type::SingleMulti,
                        width: width * 0.5,
                    });
                }
                if next_mm.len() >= WORK_SIZE {
                    let chunk = std::mem::replace(&mut next_mm, Vec::with_capacity(WORK_SIZE));
                    state.work.lock().unwrap().push(Work {
                        work: chunk,
                        ty: Type::MultiMulti,
                        width: width * 0.5,
                    });
                }
            }
            if !next_fin.is_empty() {
                state.finished.lock().unwrap().push(next_fin);
            }
            if !next_sm.is_empty() {
                state.work.lock().unwrap().push(Work {
                    work: next_sm,
                    ty: Type::SingleMulti,
                    width: width * 0.5,
                });
            }
            if !next_mm.is_empty() {
                state.work.lock().unwrap().push(Work {
                    work: next_mm,
                    ty: Type::MultiMulti,
                    width: width * 0.5,
                });
            }
        };
        let do_sm_drain = |sm: Vec<[u32; 2]>, width: Coord, state: &State| {
            let mut next_fin = Vec::with_capacity(WORK_SIZE);
            let mut next_sm = Vec::with_capacity(WORK_SIZE);

            for [p1, i2] in sm {
                // Append (_, single) coset. Single particle must be outside the split if
                // the tree is valid.
                let pos1 = particles[p1 as usize];
                if (pos1 - mids[i2 as usize]).norm() > 1.0 + 2.0 * width {
                    continue;
                }

                for i in 0..8 {
                    match Node::from_raw_index(arena[i2 as usize][i as usize]) {
                        None => continue,
                        Some(Node::Final { particle_idx: p2 }) => {
                            if (pos1 - particles[p2 as usize]).norm_squared() <= 1.0 {
                                next_fin.push((p1, p2));
                            }
                        }
                        Some(Node::Split { interm_idx: i2 }) => next_sm.push([p1, i2]),
                    }
                }
                if next_fin.len() >= WORK_SIZE {
                    let chunk = std::mem::replace(&mut next_fin, Vec::with_capacity(WORK_SIZE));
                    state.finished.lock().unwrap().push(chunk);
                }
                if next_sm.len() >= WORK_SIZE {
                    let chunk = std::mem::replace(&mut next_sm, Vec::with_capacity(WORK_SIZE));
                    state.work.lock().unwrap().push(Work {
                        work: chunk,
                        ty: Type::SingleMulti,
                        width: width * 0.5,
                    });
                }
            }
            if !next_fin.is_empty() {
                state.finished.lock().unwrap().push(next_fin);
            }
            if !next_sm.is_empty() {
                state.work.lock().unwrap().push(Work {
                    work: next_sm,
                    ty: Type::SingleMulti,
                    width: width * 0.5,
                });
            }
        };

        std::thread::scope(|scope| {
            let state = &state;
            let workers = (0..NUM_THREADS)
                .map(|i| {
                    scope.spawn(move || {
                        'outer: loop {
                            let tried = state.work.lock().unwrap().pop();
                            let Some(work) = tried else {
                                state.semaphore.fetch_sub(1, Ordering::SeqCst);
                                while state.work.lock().unwrap().is_empty() {
                                    if state.semaphore.load(Ordering::SeqCst) == 0 {
                                        break 'outer;
                                    }
                                    std::thread::yield_now();
                                }
                                state.semaphore.fetch_add(1, Ordering::SeqCst);
                                continue 'outer;
                            };
                            let _ = i;
                            /*
                            println!(
                                "Thread {i} work {} {:?} #{}",
                                work.width,
                                work.ty,
                                work.work.len()
                            );
                            */

                            match work.ty {
                                Type::MultiMulti => do_mm_drain(work.work, work.width, state),
                                Type::SingleMulti => do_sm_drain(work.work, work.width, state),
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();
            while state.semaphore.load(Ordering::SeqCst) > 0 {
                std::thread::yield_now();
            }
            for worker in workers {
                worker.join().unwrap();
            }

            assert!(state.work.lock().unwrap().is_empty());
            let mut pairs = Vec::with_capacity(particles.len() * particles.len() / 1000);
            while let Some(mut finished) = state.finished.lock().unwrap().pop() {
                pairs.append(&mut finished);
            }

            pairs
        })
    }
}
type Vectorf = Vector3<Coord>;

fn naive(particles: &[Vectorf]) -> Vec<(u32, u32)> {
    let mut pairs = Vec::with_capacity(1024 * 1024 * 1024);

    let now = Instant::now();

    for i in 0..particles.len() {
        //for j in i + 1..particles.len() {
        for j in 0..particles.len() {
            if (particles[i] - particles[j]).norm_squared() <= /*RADIUS * RADIUS*/ 1.0 {
                pairs.push((i as u32 + 1, j as u32 + 1));
            }
        }
    }

    /*for row_res in rows {
        let row = row_res.context("failed to read row")?;
    }*/

    // to measure perf
    core::hint::black_box(&pairs);

    let elapsed = now.elapsed();
    println!("Naïve: {} pairs", pairs.len());
    println!("Naïve implementation took {elapsed:?}");

    pairs.dedup();

    pairs
}

const VALIDATE: bool = false;
const CRAZY_SIMD: bool = true;
const NUM_THREADS: usize = 32;
const WORK_SIZE: usize = 1024;

fn main() -> Result<()> {
    let file = BufReader::new(File::open(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/positions.xyz".into()),
    )?);

    let now00 = Instant::now();
    let raw_particles = {
        // reserve index 0 for null sentinel
        std::iter::once(Ok(Vectorf::repeat(Coord::NAN)))
            .chain(file.lines().map(|line| -> Result<Vectorf> {
                let line = line?;
                let mut words = line.split(' ').map(|w| w.parse::<Coord>());
                Ok(Vectorf::new(
                    words.next().context("1st not present")??,
                    words.next().context("2nd not present")??,
                    words.next().context("3rd not present")??,
                ) / RADIUS)
            }))
            .collect::<Result<Vec<_>>>()?
    };
    println!("Parsed in {:?}", now00.elapsed());
    let particles = &raw_particles[1..];

    let now0 = Instant::now();
    let max_depth = {
        // Calculate size of the smallest cube that fits all points (without translating or
        // rotating the data).

        let mut max = Vectorf::zeros();
        let mut min = Vectorf::zeros();
        for particle in particles {
            for coord in 0..3 {
                if particle[coord] > max[coord] {
                    max[coord] = particle[coord]; // / RADIUS;
                } else if particle[coord] < max[coord] {
                    min[coord] = particle[coord]; // / RADIUS;
                }
            }
        }
        let min = min.map(|m| (m.floor() as i32).min(0));
        let max = max.map(|m| (m.ceil() as i32).max(0));
        println!(
            "Bounds: {} {} {}, {} {} {}",
            min[0], min[1], min[2], max[0], max[1], max[2]
        );

        max.iter()
            .copied()
            .chain(min.iter().copied())
            .map(|coord| coord.abs() as u32)
            .max()
            .unwrap_or(0)
            .next_power_of_two()
            .trailing_zeros()
    };
    let elapsed = now0.elapsed();
    println!("Tree max depth: {max_depth}, calculated in {elapsed:?}");

    let mut tree = Tree {
        root: NULL_IDX,
        root_level: max_depth as i8,
        mw: Coord::INFINITY,
        mid: Vectorf::zeros(),
        deepest: 0,
        len: 0,
    };
    let now0 = Instant::now();

    // This step is O(n log n). Even when unbalanced, this holds so long as the coordinate type
    // (int/float) is not infinitely precise (subdivisible).

    // TODO: more efficient guess?
    let mut arena = Vec::<[RawIndex; 8]>::with_capacity(2 * particles.len());
    let mut mids = Vec::<Vectorf>::with_capacity(2 * particles.len());
    // reserve null index
    arena.push([0; 8]);
    mids.push(Vectorf::repeat(Coord::NAN));

    for i in 0..particles.len() {
        tree.insert(
            NonZeroU32::new(1 + i as u32).unwrap(),
            &raw_particles,
            &mut arena,
            &mut mids,
        );
    }
    mids.push(Vectorf::repeat(Coord::NAN));
    println!("Built tree in {:?}", now0.elapsed());

    if VALIDATE {
        let mut all = Vec::new();
        Tree::validate(
            tree.root,
            tree.mid,
            Coord::powi(2.0, tree.root_level.into()) * 0.5,
            &mut all,
            &raw_particles,
            &arena,
            &mids,
        );
        all.sort_unstable();
        for i in 0..particles.len() {
            assert_eq!(all[i] as usize, i + 1);
        }
        assert_eq!(all.len(), tree.len as usize);
        println!(
            "+lvl {} -lvl {}, deepest {}, len {}",
            tree.root_level, tree.deepest, tree.mw, tree.len
        );
    }
    let _ = core::hint::black_box(&tree);

    println!("Input: {} coordinates", particles.len());

    println!("Running fast search");
    let now2 = Instant::now();
    let _ = core::hint::black_box(&arena);
    let pairs = tree.neighbor_radius_search(&raw_particles, &arena, &mids);
    let _ = core::hint::black_box(&pairs);
    println!(
        "Fast search took {:?}, {} pairs",
        now2.elapsed(),
        pairs.len()
    );

    if VALIDATE {
        let mut pairs_naive = naive(&particles);

        let mut pairs2 = Vec::new();
        for &(p, q) in &pairs {
            pairs2.push((p, q));
            pairs2.push((q, p));
        }
        pairs_naive.sort();
        pairs_naive.dedup();
        pairs2.sort();
        pairs2.dedup();

        for (p, q) in pairs_naive.iter().zip(pairs2.iter()) {
            assert_eq!(*p, *q);
        }
        assert_eq!(pairs2.len(), pairs_naive.len());
    }

    Ok(())
}
