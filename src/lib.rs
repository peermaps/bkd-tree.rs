extern crate failure;
extern crate random_access_storage;
extern crate serde_json;
extern crate serde;
extern crate serde_derive;
extern crate bincode;

use random_access_storage::RandomAccess;
use failure::Error;
use std::marker::PhantomData;
use std::mem::{size_of};
use serde_json as json;
use serde::{Serialize,Deserialize};
use serde_derive::{Serialize,Deserialize};
use bincode::{serialize};

mod bkdtree_builder;
pub use crate::bkdtree_builder::BKDTreeBuilder;

pub struct QueryResults<T> {
  _marker: PhantomData<T>,
  index: usize
}

impl<T> Iterator for QueryResults<T> {
  type Item = (T,Vec<u8>);
  fn next (&mut self) -> Option<Self::Item> {
    unimplemented!();
  }
}

pub trait Point<T> {
  fn len (self) -> usize;
  fn get (self, usize) -> T;
}
impl<T> Point<T> for [T;2] where T: Copy {
  fn len (self) -> usize { 2 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;3] where T: Copy {
  fn len (self) -> usize { 3 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;4] where T: Copy {
  fn len (self) -> usize { 4 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;5] where T: Copy {
  fn len (self) -> usize { 5 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;6] where T: Copy {
  fn len (self) -> usize { 6 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for (T,T) {
  fn len (self) -> usize { 2 }
  fn get (self, i: usize) -> T {
    match i%2 {
      0 => self.0, 1 => self.1,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T) {
  fn len (self) -> usize { 3 }
  fn get (self, i: usize) -> T {
    match i%3 {
      0 => self.0, 1 => self.1, 2 => self.2,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T) {
  fn len (self) -> usize { 4 }
  fn get (self, i: usize) -> T {
    match i%4 {
      0 => self.0, 1 => self.1, 2 => self.2, 3 => self.3,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T,T) {
  fn len (self) -> usize { 5 }
  fn get (self, i: usize) -> T {
    match i%5 {
      0 => self.0, 1 => self.1, 2 => self.2,
      3 => self.3, 4 => self.4,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T,T,T) {
  fn len (self) -> usize { 6 }
  fn get (self, i: usize) -> T {
    match i%6 {
      0 => self.0, 1 => self.1, 2 => self.2,
      3 => self.3, 4 => self.4, 5 => self.5,
      _ => panic!("impossible out of bounds")
    }
  }
}

pub struct Op<P,V,T> where P: Serialize+Copy+Point<T> {
  pub _marker: PhantomData<T>,
  pub kind: OpKind,
  pub point: P,
  pub value: V
}
pub enum OpKind { Insert, Delete }

#[derive(Serialize,Deserialize)]
struct Meta {
  mask: Vec<bool>,
  branch_factor: usize
}

impl Meta {
  fn new (branch_factor: usize) -> Self {
    Self {
      branch_factor,
      mask: vec![]
    }
  }
}

struct Staging<P,V,T> where P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  rows: Vec<Op<P,V,T>>,
  n: usize,
  count: usize,
  data: Vec<u8>
}

impl<P,V,T> Staging<P,V,T> where P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  pub fn new (n: usize, dim: usize) -> Self {
    let presize = (n+7)/8;
    let len = 4+presize+n*(size_of::<P>()+size_of::<V>());
    Self {
      rows: vec![],
      count: 0,
      n,
      data: vec![0;len]
    }
  }
  #[inline]
  pub fn size (&self) -> usize {
    self.data.len()
  }
  pub fn add (&mut self, row: Op<P,V,T>) -> Result<bool,Error> {
    println!("row:{}",row.point.len());
    if self.rows.len() == self.n {
      self.rows[self.count] = row;
    } else {
      self.rows.push(row);
    }
    self.count += 1;
    Ok(self.count >= self.n)
  }
  pub fn query (&self, bbox: (P,P)) -> Result<(),Error> {
    Ok(())
  }
  pub fn reset (&mut self) -> () {
    self.count = 0;
  }
  pub fn pack (&mut self) -> Result<(usize,&Vec<u8>),Error> {
    let size = 0;
    let psize = size_of::<P>();
    let vsize = size_of::<V>();
    for i in 0..self.count {
      let row = &self.rows[i];
      let bit = match &row.kind {
        Insert => (1 << (self.count % 8)),
        Delete => 0
      };
      let j = 4 + (i+7)/8;
      self.data[j] = self.data[j] | bit;
      let offset = 4+i*(psize+vsize);
      let pv = (&row.point,&row.value);
      let end_offset = offset+psize+vsize;
      let buf: Vec<u8> = serialize(&pv)?;
      self.data[offset..end_offset].copy_from_slice(&buf[0..]);
    }
    Ok((self.count,&self.data))
  }
}

pub struct BKDTree<S,U,P,V,T>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  branch_factor: usize,
  n: usize,
  dim: usize,
  meta_store: S,
  staging_store: S,
  open_storage: U,
  meta: Meta,
  staging: Staging<P,V,T>,
  trees: Vec<S>
}

impl<S,U,P,V,T> BKDTree<S,U,P,V,T>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  pub fn open (open_storage: U) -> Result<Self,Error> {
    let branch_factor: usize = 4;
    let n = branch_factor.pow(5);
    let dim = 2;
    let mut bkd = Self {
      meta_store: open_storage("meta")?,
      staging_store: open_storage("staging")?,
      open_storage,
      meta: Meta::new(branch_factor),
      trees: vec![],
      branch_factor: branch_factor,
      n,
      dim,
      staging: Staging::new(n,dim)
    };
    bkd.init()?;
    Ok(bkd)
  }
  fn init (&mut self) -> Result<(),Error> {
    self.meta = match self.meta_store.read(0, 1024) {
      Err(_) => Ok(Meta::new(self.branch_factor)),
      Ok(b) => json::from_slice(&b)
    }?;
    let buf = match self.staging_store.read(0, self.staging.size()) {
      Err(_) => vec![0;self.staging.size()],
      Ok(b) => b
    };
    Ok(())
  }
  pub fn builder (storage: U) -> BKDTreeBuilder<S,U,P,V,T> {
    BKDTreeBuilder::new(storage)
  }
  pub fn batch (&mut self, rows: Vec<Op<P,V,T>>) -> Result<(),Error> {
    for row in rows {
      let full = self.staging.add(row)?;
      if full {
        self.staging.reset();
      }
    };
    let (size,buf) = self.staging.pack()?;
    if size == buf.len() {
      self.staging_store.write(0,buf)?;
    } else {
      self.staging_store.write(0,&buf[0..size])?;
    }
    Ok(())
  }
  pub fn query (&mut self) -> Result<(),Error> {
    Ok(())
  }
}
