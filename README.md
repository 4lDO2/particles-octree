# Student challenge particle simulation

This repository contains details for the coding challenge announced during
student fair FARM at Chalmers, November 2024.

## Background

All IPS software relies on fast computations of collision between objects. In
our particle simulations we need to find all the pairs of particles which are
within a certain distance of each other. Such a computation needs to be highly
optimized and rely on smart algorithms to be competitive.

## Challenge

In this challenge, we provide a data set with points in space where it is your
task to find the number of pairs which are within a certain distance from each
other. You propose a solution and submit it to us.

In the below example, a particle simulation of irregular objects in an
industrial screen is shown. From this particle population, we have extracted the
positions of all particle to a single [file](data/positions.xyz) with a format
of one position per row, with x,y and z components printed with a white space
separation. There are two different sizes of the problem, where
[positions_large.xyz](data/positions_large.xyz) contains 130 000 particles.

![Picture of particles in an industrial screen.](images/screen.png?raw=true "Particle population in an industrial screen.")

Your task is to write a piece code to compute all pairs of positions which are
in a 0.05 m distance from each other. You should compute the total number of
such pairs in the provided data file with an as efficient solution as possible.

You can use your programming language of choice, but remember that the task is
about performance. To achieve optimal computational time you could also use
either CPU multithreading or a GPU parallelization.

If you choose a compiled language like C++ (which is encouraged!) or Rust please also
provide some README with instructions on building your source code or provide a
build system file.

The solution will be evaluated based on:

- computational time
- choice of algorithm, and
- creativity in optimizing your implementation.

We will test your code on a 16 core AMD 7950x desktop with a NVIDIA RTX 4080.

## Submitting the solution and questions

For any questions, please contact recruit@industrialpathsolutions.com.

To submit your solution, push a repository to a private account ot GitHub.com
(or similar service) and send a link to your repository to
recruit@industrialpathsolutions.com. All submitted contributions will be read.
