extern crate failure;
extern crate serde;
extern crate random_access_storage;

use std::marker::PhantomData;
use std::clone::Clone;
use failure::Error;
use random_access_storage::RandomAccess;
use serde::{Serialize,Deserialize};
use crate::BKDTree;

pub struct BKDTreeBuilder<S,U,P,V>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Clone+Serialize, V: Clone+Serialize {
  _marker1: PhantomData<S>,
  _marker2: PhantomData<P>,
  _marker3: PhantomData<V>,
  storage: U
}

impl<S,U,P,V> BKDTreeBuilder<S,U,P,V>
where S: RandomAccess<Error=Error>, U: (Fn(&str) -> Result<S,Error>),
P: Clone+Serialize, V: Clone+Serialize {
  pub fn new (storage: U) -> Self {
    Self {
      storage: storage,
      _marker1: PhantomData,
      _marker2: PhantomData,
      _marker3: PhantomData
    }
  }
  pub fn build (self) -> Result<BKDTree<S,U,P,V>,Error> {
    let bkd = BKDTree::open(self.storage)?;
    Ok(bkd)
  }
}
