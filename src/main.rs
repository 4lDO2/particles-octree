#![feature(let_chains)]
#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroU32;
use std::time::Instant;

use anyhow::{Context, Result};
use nalgebra::Vector3;

#[derive(Debug)]
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
    Vectorf::new(if idx & 1 == 1 { 1.0 } else { -1.0 }, if idx & 2 == 2 { 1.0 } else { -1.0 }, if idx & 4 == 4 { 1.0 } else { -1.0 })
}

impl Tree {
    fn validate(node: RawIndex, middle: Vectorf, next_gran: f32, all: &mut Vec<u32>, particles: &[Vectorf], arena: &[[RawIndex; 8]]) {
        match Node::from_raw_index(node) {
            None => (),
            Some(Node::Split { interm_idx }) => {
                let mut sub = [const { Vec::new() }; 8];
                for i in 0..8 {
                    Self::validate(arena[interm_idx as usize][i], middle + idx_to_vector(i as u8) * next_gran, next_gran * 0.5, &mut sub[i], particles, arena);
                }
                for i in 0..8 {
                    for p in &sub[i] {
                        let p = *p as usize;
                        //println!("Validating({i}) {p:?} {:?} nlvl {next_gran}", particles[p]);
                        //println!("Mid {middle:?}");
                        if i & 1 == 1 {
                            assert!(particles[p].x >= middle.x);
                        } else {
                            assert!(particles[p].x < middle.x);
                        }
                        if i & 2 == 2 {
                            assert!(particles[p].y >= middle.y);
                        } else {
                            assert!(particles[p].y < middle.y);
                        }
                        if i & 4 == 4 {
                            assert!(particles[p].z >= middle.z);
                        } else {
                            assert!(particles[p].z < middle.z);
                        }
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
            /*println!("upwards");
            // Entire tree is too small to contain both 'index' and 'self.root', must increase root
            // level.

            let child_idx = arena.len() as u32;
            arena.push([0; 8]);
            arena[child_idx as usize][usize::from(vector_to_idx(&root_shift))] = self.root;
            self.root = Node::Split {
                interm_idx: child_idx,
            }
            .raw();
            self.root_level += 1;
            return;*/
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
                            CurrentNode::Root => self.root = Node::Split { interm_idx: split_idx }.raw(),
                            CurrentNode::Split { interm, child } => arena[interm as usize][child as usize] = Node::Split { interm_idx: split_idx }.raw(),
                        }

                        arena.push([NULL_IDX; 8]);
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
            current_node = CurrentNode::Split { interm, child: child_idx as u8 };
        }

        //self.root.as_mut().unwrap().insert(&self.mid, index, , all_positions);
    }
    pub fn neighbor_radius_search(
        &self,
        particles: &[Vectorf],
        arena: &[[RawIndex; 8]],
    ) -> Vec<(u32, u32)> {
        // TODO: parallelize

        #[derive(Debug)]
        struct InProgress {
            p1: RawIndex,
            mid1: Vectorf,
            p2: RawIndex,
            mid2: Vectorf,
        }

        let mut in_progress = Vec::new();
        let mut next_in_progress = Vec::new();

        /*match Node::from_raw_index(self.root) {
            None | Some(Node::Final { .. }) => return Vec::new(),
            Some(Node::Split { interm_idx }) => {
                for i in 0..8_usize {
                    for j in 0..=i {
                        let c = |w, b| {
                            (if w & (1_usize << b) == 0_usize { -1 } else { 1 })
                                << (self.root_level - self.deepest)
                        };
                        in_progress.push(InProgress {
                            p1: arena[interm_idx as usize][i],
                            addr1: Vector3::new(c(i, 0), c(i, 1), c(i, 2)),
                            p2: arena[interm_idx as usize][j],
                            addr2: Vector3::new(c(j, 0), c(j, 1), c(j, 2)),
                        });
                    }
                }
            }
        }*/
        in_progress.push(InProgress {
            p1: self.root,
            mid1: self.mid,
            p2: self.root,
            mid2: self.mid,
        });

        let mut pairs = Vec::new();
        let mut width = f32::powi(2.0, self.root_level.into());

        let mut total_pairs = 0;
        loop {
            for InProgress {
                p1,
                mid1,
                p2,
                mid2,
            } in in_progress.drain(..)
            {
                match (Node::from_raw_index(p1), Node::from_raw_index(p2)) {
                    (None, _) | (_, None) => continue,
                    (
                        Some(Node::Split { interm_idx: i1 }),
                        Some(Node::Split { interm_idx: i2 }),
                    ) => {
                        // Append Cartesian product of respective subnodes, filtering out the ones
                        // that cannot be within range at all.

                        let diff = mid2 - mid1;
                        let diag = diff.component_div(&diff.abs()) * width * f32::sqrt(2.0);
                        if (diff + diag).norm() >= 1.0 {
                            //continue;
                        }

                        for i in 0..8 {
                            for j in 0..8 {
                                next_in_progress.push(InProgress {
                                    p1: arena[i1 as usize][i as usize],
                                    mid1: mid1 + idx_to_vector(i) * width * 0.5,
                                    p2: arena[i2 as usize][j as usize],
                                    mid2: mid2 + idx_to_vector(j) * width * 0.5,
                                });
                            }
                        }
                    }
                    (Some(Node::Final { particle_idx }), Some(Node::Split { interm_idx }))
                    | (Some(Node::Split { interm_idx }), Some(Node::Final { particle_idx })) => {
                        //println!("s{interm_idx} p{particle_idx}");
                        // Append (_, single) coset. Single particle must be outside the split if
                        // the tree is valid.
                        for i in 0..8 {
                            next_in_progress.push(InProgress {
                                p1: arena[interm_idx as usize][i as usize],
                                mid1: mid1 + idx_to_vector(i) * width * 0.5,
                                p2: particle_idx as i32,
                                mid2, // ignored from now
                            });
                        }
                    }
                    (
                        Some(Node::Final { particle_idx: f1 }),
                        Some(Node::Final { particle_idx: f2 }),
                    ) => {
                        //println!("f{f1} f{f2}");
                        total_pairs += 1;
                        if (particles[f1 as usize] - particles[f2 as usize]).norm_squared() <= 1.0 {
                            pairs.push((f1, f2));
                        }
                    }
                }
            }
            width *= 0.5;
            if next_in_progress.is_empty() {
                assert!(in_progress.is_empty());
                break;
            }
            std::mem::swap(&mut in_progress, &mut next_in_progress);
        }

        //pairs.dedup();
        dbg!(total_pairs);
        pairs
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
                pairs.push((i as u32, j as u32));
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

fn main() -> Result<()> {
    let file = BufReader::new(File::open(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "data/positions.xyz".into()),
    )?);

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
    let particles = &raw_particles[1..];

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
            //+ 1
    };
    println!("Tree max depth: {max_depth}");

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
    // reserve null index
    arena.push([0; 8]);

    for i in 0..particles.len() {
        //println!("Inserting {}: {:?}", i + 1, particles[i]);
        tree.insert(
            NonZeroU32::new(1 + i as u32).unwrap(),
            &raw_particles,
            &mut arena,
        );
        //println!("Tree: {tree:?}");
        //println!("Arena: {:?}", &arena[1..]);
    }
    //Tree::validate(tree.root, tree.mid, f32::powi(2.0, tree.root_level.into()) * 0.5, &mut Vec::new(), &raw_particles, &arena);
    println!("Built tree in {:?}", now0.elapsed());
    let mut all = Vec::new();
    Tree::validate(tree.root, tree.mid, f32::powi(2.0, tree.root_level.into()) * 0.5, &mut all, &raw_particles, &arena);
    all.sort_unstable();
    for i in 0..particles.len() {
        assert_eq!(all[i] as usize, i + 1);
    }
    assert_eq!(all.len(), tree.len as usize);
    println!(
        "+lvl {} -lvl {}, {}, len {}",
        tree.root_level, tree.deepest, tree.mw, tree.len
    );
    //println!("Tree: {tree:?}");
    core::hint::black_box(&tree);

    println!("Input: {} coordinates", particles.len());

    println!("Running fast search");
    let now2 = Instant::now();
    let _ = core::hint::black_box(&arena);
    let pairs = tree.neighbor_radius_search(&raw_particles, &arena);
    let _ = core::hint::black_box(&pairs);
    println!("Fast search took {:?}, {} pairs", now2.elapsed(), pairs.len());

    let pairs_naive = naive(&particles);
    assert_eq!(pairs.len(), pairs_naive.len());

    Ok(())
}
