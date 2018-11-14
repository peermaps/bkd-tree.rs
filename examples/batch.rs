extern crate bkd_tree;
extern crate random_access_disk;
extern crate failure;
extern crate rand;

use std::path::PathBuf;
use random_access_disk::RandomAccessDisk;
use failure::Error;
use bkd_tree::{BKDTree,Row};
use rand::random;

fn main () -> Result<(),Error> {
  let mut bkd = BKDTree::open(storage)?;
  let batch = (0..100).map(|_| {
    let lon: f32 = (random::<f32>()*2.0-1.0)*180.0;
    let lat: f32 = (random::<f32>()*2.0-1.0)*90.0;
    let id = random::<u32>();
    Row::insert([lon,lat],id)
  }).collect();
  bkd.batch(batch)?;

  let args: Vec<String> = std::env::args().collect();
  let bbox = (
    [args[1].parse()?,args[2].parse()?],
    [args[3].parse()?,args[4].parse()?]
  );

  for result in bkd.query(bbox) {
    println!("{:?}", result);
  }
  Ok(())
}

fn storage (name:&str) -> Result<RandomAccessDisk,Error> {
  let mut p = PathBuf::from("/tmp/db/");
  p.push(name);
  Ok(RandomAccessDisk::open(p)?)
}
