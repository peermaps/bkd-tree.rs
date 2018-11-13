extern crate failure;
extern crate random_access_storage;
extern crate serde;

use std::marker::PhantomData;
use failure::Error;
use random_access_storage::RandomAccess;
use crate::{BKDTree,Point};
use serde::Serialize;

pub struct BKDTreeBuilder<S,U,P,V,T>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  _marker1: PhantomData<S>,
  _marker2: PhantomData<P>,
  _marker3: PhantomData<V>,
  _marker4: PhantomData<T>,
  storage: U
}

impl<S,U,P,V,T> BKDTreeBuilder<S,U,P,V,T>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Serialize+Copy+Point<T>, V: Serialize+Copy {
  pub fn new (storage: U) -> Self {
    Self {
      _marker1: PhantomData,
      _marker2: PhantomData,
      _marker3: PhantomData,
      _marker4: PhantomData,
      storage: storage
    }
  }
  pub fn build (self) -> Result<BKDTree<S,U,P,V,T>,Error> {
    let bkd = BKDTree::open(self.storage)?;
    Ok(bkd)
  }
}
