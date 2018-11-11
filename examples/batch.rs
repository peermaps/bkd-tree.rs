extern crate bkd_tree;
extern crate random_access_disk;
extern crate failure;

use std::path::PathBuf;
use random_access_disk::RandomAccessDisk;
use failure::Error;
use bkd_tree::{BKDTree,Op,OpKind::Insert};

fn main () -> Result<(),Error> {
  let mut bkd = BKDTree::open(storage)?;
  bkd.batch(vec![
    Op { kind: Insert, point: (5.55f32,6.66f32), value: 1234u32 }
  ])?;
  Ok(())
}

fn storage (name:&str) -> Result<RandomAccessDisk,Error> {
  let mut p = PathBuf::from("/tmp/db/");
  p.push(name);
  let ra = RandomAccessDisk::open(p)?;
  Ok(ra)
}
