#![feature(let_chains)]

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::{Context, Result};
use nalgebra::Vector3;

enum Node {
    Split { children: Box<[Option<Node>; 8]> },
    Final { particle_idx: u32 },
}
struct Tree {
    root: Option<Node>,
    root_level: i8,
}

const RADIUS: f32 = 0.05;

fn vector_to_idx(vector: &Vectorf) -> u8 {
    u8::from(vector.x >= 0.0) | (u8::from(vector.y >= 0.0) << 1) | (u8::from(vector.z >= 0.0) << 2)
}

impl Tree {
    pub fn insert(&mut self, index: u32, all_positions: &[Vectorf]) {
        if self.root.is_none() || pos_radius_relative.magnitude_squared() >= 3.0 * f32::powi(2.0, self.root_level.into()) {
            let mut children = Box::new([const { None }; 8]);
            children[usize::from(vector_to_idx(pos_radius_relative))] = self.root.take();
            let parent = Node::Split { children };
            self.root = Some(parent);
            self.root_level += 1;
        }
        self.root.as_mut().unwrap().insert(pos_radius_relative, index, self.root_level);
    }
}
impl Node {
    pub fn insert(&mut self, mid: &Vectorf, index: u32, level: i8, positions: &[Vectorf]) {
        let children = match *self {
            Node::Final { particle_idx } => if particle_idx == index {
                return;
            } else {
                *self = Node::Split { children: Default::default() };
                let Node::Split { ref mut children } = self else { unreachable!() };
                children
            }
            Node::Split { ref mut children } => children,
        };

        let aabb_size = f32::powi(2.0, level.into());
        let child_idx = usize::from(vector_to_idx(&pos_radius_relative));
        let child = &mut children[child_idx];

        if let Some(child) = child {
            let direction = Vectorf::new(
                if child_idx & 1 == 1 { 1.0 } else { -1.0 },
                if child_idx & 2 == 2 { 1.0 } else { -1.0 },
                if child_idx & 4 == 4 { 1.0 } else { -1.0 },
            );
            child.insert(&(pos_radius_relative + direction * aabb_size), index, level - 1)
        } else {
            *child = Some(Node::Final { particle_idx: index });
        }
    }
}

type Vectorf = Vector3<f32>;

fn square(v: Vectorf) -> f32 {
    v.dot(&v)
}

fn main() -> Result<()> {
    let file = BufReader::new(File::open(
        std::env::args()
            .nth(1)
            .context("expected input file as argument")?,
    )?);

    let particles = {
        file.lines().map(|line| -> Result<Vectorf> {
            let line = line?;
            let mut words = line.split(' ').map(|w| w.parse::<f32>());
            Ok(Vectorf::new(
                words.next().context("1st not present")??,
                words.next().context("2nd not present")??,
                words.next().context("3rd not present")??,
            ))
        }).collect::<Result<Vec<_>>>()?
    };

    let max_depth = {
        // Calculate size of the smallest cube that fits all points (without translating or
        // rotating the data).

        let mut max = Vectorf::zeros();
        let mut min = Vectorf::zeros();
        for particle in &particles {
            for coord in 0..3 {
                if particle[coord] > max[coord] {
                    max[coord] = particle[coord] / RADIUS;
                } else if particle[coord] < max[coord] {
                    min[coord] = particle[coord] / RADIUS;
                }
            }
        }
        let min = min.map(|m| (m.floor() as i32).min(0));
        let max = max.map(|m| (m.ceil() as i32).max(0));
        println!("Bounds: {} {} {}, {} {} {}", min[0], min[1], min[2], max[0], max[1], max[2]);

        max.iter().copied().chain(min.iter().copied())
            .map(|coord| coord.abs() as u32)
            .max().unwrap_or(0).next_power_of_two().trailing_zeros() + 1

    };
    println!("Tree max depth: {max_depth}");

    let mut tree = Tree {
        root: None,
        root_level: 0,
    };
    for i in 0..particles.len() {
        tree.insert(i as u32, &particles);
    }

    println!("Input: {} coordinates", particles.len());
    let mut pairs = Vec::with_capacity(1024 * 1024 * 1024);

    let now = Instant::now();

    for i in 0..particles.len() {
        for j in i + 1..particles.len() {
            if square(particles[i] - particles[j]) <= RADIUS * RADIUS {
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

    Ok(())
}
