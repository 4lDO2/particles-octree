use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use anyhow::{Context, Result};

fn square(v: [f32; 3]) -> f32 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}
fn subtract(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn main() -> Result<()> {
    let file = BufReader::new(File::open(
        std::env::args()
            .nth(1)
            .context("expected input file as argument")?,
    )?);

    let rows = {
        file.lines().map(|line| -> Result<[f32; 3]> {
            let line = line?;
            let mut words = line.split(' ').map(|w| w.parse::<f32>());
            Ok([
                words.next().context("1st not present")??,
                words.next().context("2nd not present")??,
                words.next().context("3rd not present")??,
            ])
        })
    };

    let threshold = 0.05;

    let particles = rows.collect::<Result<Vec<_>>>()?;
    let mut pairs = Vec::with_capacity(particles.len() * (particles.len() - 1) / 2);
    println!("Input: {} coordinates", particles.len());

    let now = Instant::now();

    for i in 0..particles.len() {
        for j in i + 1..particles.len() {
            if square(subtract(particles[i], particles[j])) <= threshold * threshold {
                pairs.push((i, j));
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
