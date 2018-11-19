extern crate bkd_tree;
extern crate random_access_disk;
extern crate failure;
extern crate rand;

use std::path::PathBuf;
use random_access_disk::RandomAccessDisk;
use failure::{Error,bail};
use bkd_tree::{BKDTree,Row};
use rand::random;

fn main () -> Result<(),Error> {
  let mut bkd = BKDTree::open(storage)?;
  let args: Vec<String> = std::env::args().collect();
  if args.len() < 2 { return bail!("must provide a command") }

  if args[1] == "populate" {
    if args.len() < 3 { return bail!("populate requires a number") }
    let n = args[2].parse()?;
    let batch = (0..n).map(|_| {
      let lon: f32 = (random::<f32>()*2.0-1.0)*180.0;
      let lat: f32 = (random::<f32>()*2.0-1.0)*90.0;
      let id = random::<u32>();
      Row::insert([lon,lat],id)
    }).collect();
    bkd.batch(batch)?;
  } else if args[1] == "query" {
    if args.len() < 6 {
      return bail!("query requires 4 bounds (w s e n)");
    }
    let bbox = (
      [args[2].parse()?,args[3].parse()?],
      [args[4].parse()?,args[5].parse()?]
    );
    for (point,value) in bkd.query(bbox) {
      println!("{{ point: {:?}, value: {:?} }}", point, value);
    }
  }
  Ok(())
}

fn storage (name:&str) -> Result<RandomAccessDisk,Error> {
  let mut p = PathBuf::from("/tmp/db/");
  p.push(name);
  Ok(RandomAccessDisk::open(p)?)
}
