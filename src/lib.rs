extern crate failure;
extern crate random_access_storage;
extern crate serde_json;
extern crate serde;
extern crate serde_derive;
extern crate byteorder;
extern crate bincode;

use random_access_storage::RandomAccess;
use failure::Error;
use std::marker::PhantomData;
use std::mem::size_of;
use serde_json as json;
use serde_derive::{Serialize,Deserialize};
use serde::{Serialize,Deserialize};
use std::clone::Clone;
use bincode::{serialize_into, deserialize};

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

#[derive(Clone)]
pub struct Op<P,V> {
  pub kind: OpKind,
  pub point: P,
  pub value: V
}
#[derive(Clone)]
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

#[derive(Serialize,Deserialize)]
struct Staging<P,V> where P: Clone+Serialize, V: Clone+Serialize {
  count: usize,
  mask: Vec<bool>,
  rows: Vec<(P,V)>
}

impl<P,V> Staging<P,V> where P: Clone+Serialize, V: Clone+Serialize {
  pub fn new (n: usize) -> Self {
    Self {
      rows: vec![],
      mask: vec![false;n],
      count: 0
    }
  }
  pub fn add (&mut self, kind: OpKind, row: (P,V)) {
    self.rows[self.count] = row;
    self.count += 1;
  }
  pub fn query (&mut self, bbox: (P,P)) -> Result<(),Error> {
    Ok(())
  }
  pub fn pack (&mut self, out: Vec<u8>) -> Result<(),Error> {
    serialize_into(out,self)?;
    Ok(())
  }
}

pub struct BKDTree<S,U,P,V>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Clone+Serialize, V: Clone+Serialize {
  _marker1: PhantomData<P>,
  _marker2: PhantomData<V>,
  branch_factor: usize,
  n: usize,
  meta_store: S,
  staging_store: S,
  open_storage: U,
  meta: Meta,
  staging: Staging<P,V>,
  trees: Vec<S>
}

impl<S,U,P,V> BKDTree<S,U,P,V>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Clone+Serialize, V: Clone+Serialize {
  pub fn open (open_storage: U) -> Result<Self,Error> {
    let branch_factor: usize = 4;
    let n = branch_factor.pow(5);
    let mut bkd = Self {
      _marker1: PhantomData,
      _marker2: PhantomData,
      meta_store: open_storage("meta")?,
      staging_store: open_storage("staging")?,
      open_storage,
      meta: Meta::new(branch_factor),
      trees: vec![],
      branch_factor: branch_factor,
      n,
      staging: Staging::new(n)
    };
    bkd.init()?;
    Ok(bkd)
  }
  fn init (&mut self) -> Result<(),Error> {
    let presize = (self.n+7)/8;
    let len = 4+presize+self.n*(size_of::<P>()+size_of::<V>());
    self.meta = match self.meta_store.read(0, 1024) {
      Err(_) => Ok(Meta::new(self.branch_factor)),
      Ok(b) => json::from_slice(&b)
    }?;
    let buf = match self.staging_store.read(0, len) {
      Err(_) => vec![0;len],
      Ok(b) => b
    };
    Ok(())
  }
  pub fn builder (storage: U) -> BKDTreeBuilder<S,U,P,V> {
    BKDTreeBuilder::new(storage)
  }
  pub fn batch (&mut self, rows: Vec<Op<P,V>>) -> Result<(),Error> {
    Ok(())
  }
  pub fn query (&mut self) -> Result<(),Error> {
    Ok(())
  }
}
