extern crate bkd_tree;
extern crate random_access_disk;
extern crate failure;
extern crate rand;

use std::path::PathBuf;
use random_access_disk::RandomAccessDisk;
use failure::Error;
use bkd_tree::{BKDTree,Op,OpKind::Insert};
use rand::random;
use std::marker::PhantomData;

fn main () -> Result<(),Error> {
  let mut bkd = BKDTree::open(storage)?;
  for _ in 0..100 {
    let lon: f32 = (random::<f32>()*2.0-1.0)*180.0;
    let lat: f32 = (random::<f32>()*2.0-1.0)*90.0;
    let id = random::<u32>();
    bkd.batch(vec![
      Op { _marker: PhantomData, kind: Insert, point: [lon,lat], value: id }
    ])?;
  }
  Ok(())
}

fn storage (name:&str) -> Result<RandomAccessDisk,Error> {
  let mut p = PathBuf::from("/tmp/db/");
  p.push(name);
  let ra = RandomAccessDisk::open(p)?;
  Ok(ra)
}
