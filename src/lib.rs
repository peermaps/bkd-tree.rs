extern crate failure;
extern crate random_access_storage;
extern crate serde_json;
extern crate serde;
extern crate serde_derive;
extern crate bincode;

use random_access_storage::RandomAccess;
use failure::Error;
use std::marker::PhantomData;
use std::mem::size_of;
use std::fmt::Debug;
use serde_json as json;
use serde::{Serialize,de::DeserializeOwned};
use serde_derive::{Serialize,Deserialize};
use bincode::{serialize,deserialize};

mod bkdtree_builder;
pub use crate::bkdtree_builder::BKDTreeBuilder;

#[derive(Debug)]
pub struct QueryResults<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy+'static,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  staging_index: usize,
  bkd: BKDTree<S,U,P,V,T>,
  deletes: Vec<(P,V)>,
  bbox: (P,P)
}

impl<S,U,P,V,T> QueryResults<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy+'static,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  fn new (bkd: BKDTree<S,U,P,V,T>, bbox: (P,P)) -> Self {
    Self {
      bkd,
      bbox,
      deletes: vec![],
      staging_index: 0
    }
  }
}

impl<S,U,P,V,T> Iterator for QueryResults<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy+'static,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  type Item = (P,V);
  fn next (&mut self) -> Option<Self::Item> {
    while self.staging_index < self.bkd.staging.count {
      let row = &self.bkd.staging.rows[self.staging_index];
      self.staging_index += 1;
      if contains(self.bbox.0, self.bbox.1, row.point) {
        match &row.kind {
          RowKind::Insert => {
            return Some((row.point,row.value))
          },
          RowKind::Delete => {
            self.deletes.push((row.point,row.value));
          }
        };
      }
    }
    None
  }
}

pub trait Point<T> where T: PartialOrd {
  fn len (self) -> usize;
  fn get (self, usize) -> T;
}
impl<T> Point<T> for [T;2] where T: Copy+PartialOrd {
  fn len (self) -> usize { 2 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;3] where T: Copy+PartialOrd {
  fn len (self) -> usize { 3 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;4] where T: Copy+PartialOrd {
  fn len (self) -> usize { 4 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;5] where T: Copy+PartialOrd {
  fn len (self) -> usize { 5 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for [T;6] where T: Copy+PartialOrd {
  fn len (self) -> usize { 6 }
  fn get (self, i: usize) -> T { self[i] }
}
impl<T> Point<T> for (T,T) where T: PartialOrd {
  fn len (self) -> usize { 2 }
  fn get (self, i: usize) -> T {
    match i%2 {
      0 => self.0, 1 => self.1,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T) where T: PartialOrd {
  fn len (self) -> usize { 3 }
  fn get (self, i: usize) -> T {
    match i%3 {
      0 => self.0, 1 => self.1, 2 => self.2,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T) where T: PartialOrd {
  fn len (self) -> usize { 4 }
  fn get (self, i: usize) -> T {
    match i%4 {
      0 => self.0, 1 => self.1, 2 => self.2, 3 => self.3,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T,T) where T: PartialOrd {
  fn len (self) -> usize { 5 }
  fn get (self, i: usize) -> T {
    match i%5 {
      0 => self.0, 1 => self.1, 2 => self.2,
      3 => self.3, 4 => self.4,
      _ => panic!("impossible out of bounds")
    }
  }
}
impl<T> Point<T> for (T,T,T,T,T,T) where T: PartialOrd {
  fn len (self) -> usize { 6 }
  fn get (self, i: usize) -> T {
    match i%6 {
      0 => self.0, 1 => self.1, 2 => self.2,
      3 => self.3, 4 => self.4, 5 => self.5,
      _ => panic!("impossible out of bounds")
    }
  }
}

#[derive(Debug)]
pub struct Row<P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
T: Debug+PartialOrd {
  _marker: PhantomData<T>,
  pub kind: RowKind,
  pub point: P,
  pub value: V
}
#[derive(Debug,PartialEq)]
pub enum RowKind { Insert, Delete }

impl<P,V,T> Row<P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
T: Debug+PartialOrd {
  pub fn insert (point: P, value: V) -> Self {
    Self { _marker: PhantomData, kind: RowKind::Insert, point, value }
  }
  pub fn delete (point: P, value: V) -> Self {
    Self { _marker: PhantomData, kind: RowKind::Delete, point, value }
  }
}

#[derive(Debug,Serialize,Deserialize)]
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
  fn save<S,U> (&self, store: &mut S) -> Result<(),Error> where
  S: Debug+RandomAccess<Error=Error>,
  U: (Fn(&str) -> Result<S,Error>) {
    let mut buf: Vec<u8> = vec![0x20;1024];
    buf[1023] = 0x0a;
    let jbuf = json::to_vec(self)?;
    buf[0..jbuf.len()].copy_from_slice(&jbuf[0..]);
    store.write(0, &buf)?;
    Ok(())
  }
}

#[derive(Debug)]
struct Staging<P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy,
T: Debug+PartialOrd {
  rows: Vec<Row<P,V,T>>,
  n: usize,
  count: usize,
  data: Vec<u8>
}

impl<P,V,T> Staging<P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy+'static,
T: Debug+PartialOrd+'static {
  pub fn new (n: usize) -> Self {
    let presize = (n+7)/8;
    let len = 4+presize+n*(size_of::<P>()+size_of::<V>());
    Self {
      rows: Vec::with_capacity(n),
      count: 0,
      n,
      data: vec![0;len]
    }
  }
  pub fn load (&mut self, buf: Vec<u8>) -> Result<(),Error>
  where P: DeserializeOwned, V: DeserializeOwned {
    let ucount: u32 = deserialize(&buf[0..4])?;
    self.count = ucount as usize;
    self.rows.truncate(0);
    let presize = (self.n+7)/8;
    let size = size_of::<P>() + size_of::<V>();
    for i in 0..self.count {
      let j = 4 + presize + i*size;
      let is_insert = ((buf[4+i/8] >> (i%8)) & 1) == 1;
      let (point,value): (P,V) = deserialize(&buf[j..j+size])?;
      self.rows.push(match is_insert {
        true => Row::insert(point, value),
        false => Row::delete(point, value)
      });
    }
    Ok(())
  }
  #[inline]
  pub fn size (&self) -> usize {
    self.data.len()
  }
  pub fn add (&mut self, row: Row<P,V,T>) -> Result<bool,Error> {
    if self.rows.len() == self.n {
      self.rows[self.count] = row;
    } else {
      self.rows.push(row);
    }
    self.count += 1;
    Ok(self.count >= self.n)
  }
  pub fn reset (&mut self) -> () {
    self.count = 0;
  }
  pub fn pack (&mut self) -> Result<(usize,&Vec<u8>),Error> {
    let ucount = self.count as u32;
    let buf: Vec<u8> = serialize(&ucount)?;
    self.data[0..4].copy_from_slice(&buf[0..]);
    let psize = size_of::<P>();
    let vsize = size_of::<V>();
    let presize = (self.n+7)/8;
    for i in 0..self.count {
      let row = &self.rows[i];
      let bit = match &row.kind {
        RowKind::Insert => (1 << (self.count % 8)),
        RowKind::Delete => 0
      };
      let j = 4 + (i+7)/8;
      self.data[j] = self.data[j] | bit;
      let offset = 4+presize+i*(psize+vsize);
      let pv = (&row.point,&row.value);
      let end_offset = offset+psize+vsize;
      let buf: Vec<u8> = serialize(&pv)?;
      self.data[offset..end_offset].copy_from_slice(&buf[0..]);
    }
    //Ok((4+presize+self.count*(psize+vsize),&self.data))
    Ok((self.data.len(),&self.data))
  }
}

#[derive(Debug,Clone,Copy)]
struct Tree<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  _marker0: PhantomData<P>,
  _marker1: PhantomData<V>,
  _marker2: PhantomData<T>,
  _marker3: PhantomData<U>,
  storage: S,
  branch_factor: usize,
  size: usize,
  bytes: usize,
  n: usize,
  presize: usize,
  row_size: usize
}

impl<S,U,P,V,T> Tree<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  pub fn new (storage: S, branch_factor: usize, i: usize, n: usize) -> Self {
    let presize = (n+7)/8;
    let size = n*2usize.pow(i as u32);
    let row_size = size_of::<P>() + size_of::<V>();
    Self {
      _marker0: PhantomData,
      _marker1: PhantomData,
      _marker2: PhantomData,
      _marker3: PhantomData,
      storage,
      branch_factor,
      size,
      bytes: presize + size*row_size,
      n,
      presize,
      row_size
    }
  }
  pub fn copy_into (&self, out: &mut Vec<&Row<P,V,T>>) -> Result<(),Error> {
    // TODO: walk the tree and push rows to `out`
    Ok(())
  }
  fn build_walk (&mut self, buf: &mut Vec<u8>, rows: &mut [&Row<P,V,T>],
  depth: usize, index: usize) -> Result<(),Error> {
    let B = self.branch_factor;
    if rows.len() == 1 {
      self.build_write(buf, index, rows[0])?;
    }
    if rows.len() <= 1 { return Ok(()) }
    let dim = rows[0].point.len();
    let axis = depth % dim;
    rows.sort_unstable_by(|a,b| {
      let pa = a.point.get(axis);
      let pb = b.point.get(axis);
      match pa.partial_cmp(&pb) {
        Some(x) => x,
        None => std::cmp::Ordering::Less
      }
    });
    let mut j = 0;
    let mut n = 0;
    let mut pk = std::usize::MAX;
    let len = rows.len() as f32;
    let step = len / (B as f32);
    let mut i = step;
    while i < len {
      let k = i as usize;
      if k == pk { break };
      pk = k;
      self.build_write(buf, index+n, rows[k])?;
      self.build_walk(buf, &mut rows[j..k],
        depth+1, Self::calc_index(B, index, n))?;
      j = k + 1;
      n += 1;
      i += step;
   }
    self.build_walk(buf, &mut rows[j..],
      depth+1, Self::calc_index(B, index, n))?;
    Ok(())
  }
  fn calc_index (B: usize, index: usize, n: usize) -> usize {
    index * B + (B-1)*(n+1)
  }
  fn build_write (&mut self, buf: &mut Vec<u8>, index: usize,
  row: &Row<P,V,T>) -> Result<(),Error> {
    let j = index/8;
    buf[j] = buf[j] | (1 << (index % 8));
    let pv = (&row.point,&row.value);
    let i = self.presize + index*self.row_size;
    (&mut buf[i..i+self.row_size]).copy_from_slice(&serialize(&pv)?[0..]);
    Ok(())
  }
  fn write (&mut self, index: usize, buf: Vec<u8>) -> Result<(),Error> {
    // TODO: cache writes integrated with the LRU
    self.storage.write(index, &buf)?;
    Ok(())
  }
  fn flush (&mut self, buf: &Vec<u8>) -> Result<(),Error> {
    self.storage.write(0, buf)
  }
  pub fn build (&mut self, rows: &mut Vec<&Row<P,V,T>>) -> Result<(),Error> {
    let mut buf = vec![0;self.bytes];
    self.build_walk(&mut buf, rows, 0, 0)?;
    self.flush(&buf);
    Ok(())
  }
}

#[derive(Debug)]
pub struct BKDTree<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  branch_factor: usize,
  n: usize,
  dim: usize,
  meta_store: S,
  staging_store: S,
  open_storage: U,
  meta: Meta,
  staging: Staging<P,V,T>,
  trees: Vec<Tree<S,U,P,V,T>>
}

impl<S,U,P,V,T> BKDTree<S,U,P,V,T> where
P: Debug+Serialize+Copy+Point<T>+'static,
V: Debug+Serialize+Copy+'static,
T: Debug+PartialOrd+'static,
S: Debug+RandomAccess<Error=Error>,
U: (Fn(&str) -> Result<S,Error>) {
  pub fn open (open_storage: U) -> Result<Self,Error>
  where P: DeserializeOwned, V: DeserializeOwned {
    let branch_factor: usize = 4;
    let n = branch_factor.pow(5);
    let mut bkd = Self {
      meta_store: open_storage("meta")?,
      staging_store: open_storage("staging")?,
      open_storage,
      meta: Meta::new(branch_factor),
      trees: vec![],
      branch_factor: branch_factor,
      n,
      dim: 0,
      staging: Staging::new(n)
    };
    bkd.init()?;
    Ok(bkd)
  }
  fn init (&mut self) -> Result<(),Error>
  where P: DeserializeOwned, V: DeserializeOwned {
    self.meta = match self.meta_store.read(0, 1024) {
      Err(_) => {
        let meta = Meta::new(self.branch_factor);
        meta.save::<S,U>(&mut self.meta_store)?;
        Ok(meta)
      },
      Ok(b) => json::from_slice(&b)
    }?;
    for i in 0..self.meta.mask.len() {
      self.trees.push(Tree::new(
        (self.open_storage)(&format!("tree{}", i))?,
        self.branch_factor,
        i,
        self.n
      ));
    }
    let buf = match self.staging_store.read(0, self.staging.size()) {
      Err(_) => vec![0;self.staging.size()],
      Ok(b) => b
    };
    self.staging.load(buf)?;
    Ok(())
  }
  fn merge (&mut self) -> Result<(),Error> {
    let i = match (&self.meta.mask).into_iter().enumerate()
    .skip_while(|(_,m)| **m).next() {
      Some((i,_)) => i,
      None => {
        let i = self.meta.mask.len();
        self.meta.mask.push(false);
        self.trees.push(Tree::new(
          (self.open_storage)(&format!("tree{}", i))?,
          self.branch_factor,
          i,
          self.n
        ));
        i
      }
    };
    let last = self.trees.len()-1;
    let mut rows = Vec::with_capacity(self.trees[last].size);
    for row in &self.staging.rows { rows.push(row) }
    for i in 0..self.trees.len() {
      self.trees[i].copy_into(&mut rows)?;
    }
    self.trees[last].build(&mut rows)?;

    for j in 0..i {
      self.meta.mask[j] = false;
    }
    self.meta.save::<S,U>(&mut self.meta_store)?;
    Ok(())
  }
  pub fn builder (storage: U) -> BKDTreeBuilder<S,U,P,V,T> {
    BKDTreeBuilder::new(storage)
  }
  pub fn batch (&mut self, rows: Vec<Row<P,V,T>>) -> Result<(),Error> {
    for row in rows {
      let full = self.staging.add(row)?;
      if full {
        self.merge()?;
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
  pub fn query (self, bbox: (P,P)) -> QueryResults<S,U,P,V,T> {
    QueryResults::new(self, bbox)
  }
}

fn contains<P: Point<T>+Copy, T: PartialOrd> (min: P, max: P, point: P) -> bool
where T: Debug+PartialOrd {
  for i in 0..point.len() {
    let xmin = min.get(i);
    let xmax = max.get(i);
    let x = point.get(i);
    if x > xmax || x < xmin { return false }
  }
  true
}
