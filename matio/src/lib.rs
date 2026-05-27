use matio_sys::{
    Mat_Close, Mat_Open, Mat_Rewind, Mat_VarFree, Mat_VarGetSize, Mat_VarRead, Mat_VarReadNext,
    mat_acc, mat_acc_MAT_ACC_RDONLY, mat_acc_MAT_ACC_RDWR, mat_t, matio_types_MAT_T_DOUBLE,
    matio_types_MAT_T_SINGLE, matio_types_MAT_T_UNKNOWN, matvar_t,
};
use std::ffi::{CStr, CString, NulError, c_int};
use std::mem::size_of;
use std::path::Path;
use std::ptr::NonNull;
use std::slice;
use std::str::Utf8Error;
use thiserror::Error;
use tracing::{Level, event};

pub use matio_sys as sys;

const _: () = assert!(std::mem::size_of::<c_int>() == std::mem::size_of::<i32>());

/// Opens a MATLAB MAT file.
pub fn open(matname: impl AsRef<Path>, acc: Access) -> Result<Mat, Error> {
    Mat::open(matname, acc)
}

/// A MATLAB MAT file.
#[derive(Debug)]
pub struct Mat(NonNull<mat_t>);

impl Mat {
    /// Opens a MATLAB MAT file.
    pub fn open(matname: impl AsRef<Path>, acc: Access) -> Result<Self, Error> {
        let matname = path_to_cstring(matname.as_ref())?;
        let ptr = unsafe { Mat_Open(matname.as_ptr(), mat_acc::from(acc) as _) };
        Ok(Self(NonNull::new(ptr).ok_or(Error::OpenFailed)?))
    }

    /// Get a variable from the file.
    ///
    /// If the variable is not present on the file `Error::NotFound` is returned,
    /// we don't use `Option` here for this as the `matio` API doesn't have a
    /// way of signaling if the variable was truly not found or an error
    /// happened when reading the file.
    pub fn get(&self, name: &str) -> Result<Var, Error> {
        let cstring = CString::new(name)?;
        let ptr = unsafe { Mat_VarRead(self.0.as_ptr(), cstring.as_ptr()) };
        unsafe { Var::from_ptr(ptr).ok_or(Error::NotFound) }
    }

    /// Returns an iterator over the variables.
    pub fn vars(&mut self) -> VarIter<'_> {
        VarIter(self)
    }

    /// Returns the raw pointer.
    pub fn as_ptr(&mut self) -> *mut mat_t {
        self.0.as_ptr()
    }
}

impl Drop for Mat {
    fn drop(&mut self) {
        unsafe { Mat_Close(self.0.as_ptr()) };
    }
}

/// Iterator over variables.
#[derive(Debug)]
pub struct VarIter<'a>(&'a mut Mat);

impl<'a> Iterator for VarIter<'a> {
    type Item = Var;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe { Var::from_ptr(Mat_VarReadNext(self.0.as_ptr())) }
    }
}

impl<'a> Drop for VarIter<'a> {
    fn drop(&mut self) {
        // NOTE: Rewind back so that vars() method can start from
        // the beginning of the file.
        //
        // XXX: We don't check for errors here but what would be the
        // approiate thing to do here?
        unsafe { Mat_Rewind(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
pub struct Var(NonNull<matvar_t>);

impl Var {
    pub unsafe fn from_ptr(ptr: *mut matvar_t) -> Option<Var> {
        NonNull::new(ptr).map(Self)
    }

    /// The name of the variable.
    pub fn name(&self) -> Result<Option<&str>, Utf8Error> {
        let ptr = unsafe { (*self.0.as_ptr()).name };
        if ptr.is_null() {
            return Ok(None);
        }

        let cstr = unsafe { CStr::from_ptr(ptr) };
        cstr.to_str().map(Some)
    }

    /// The dimensions array.
    pub fn dims(&self) -> &[i32] {
        let ptr = unsafe { (*self.0.as_ptr()).dims };
        if ptr.is_null() {
            event!(
                Level::ERROR,
                var_ptr = self.0.as_ptr() as usize,
                "dims ptr is NULL"
            );
            return &[];
        }

        let rank = unsafe { (*self.0.as_ptr()).rank };
        if rank < 2 {
            event!(Level::ERROR, "rank is less than two");
            return &[];
        }

        unsafe { slice::from_raw_parts(ptr as *const i32, rank as usize) }
    }

    /// The value of the variable.
    pub fn value(&self) -> Result<Value<'_>, Error> {
        let data_type = unsafe { (*self.0.as_ptr()).data_type };

        #[allow(non_upper_case_globals)]
        match data_type {
            matio_types_MAT_T_UNKNOWN => Ok(Value::Unknown),
            matio_types_MAT_T_SINGLE => Ok(Value::Single(unsafe { self.data::<f32>().unwrap() })),
            matio_types_MAT_T_DOUBLE => Ok(Value::Double(unsafe { self.data::<f64>().unwrap() })),
            _ => unimplemented!("data_type = {data_type}"),
        }
    }

    unsafe fn data<'a, T>(&'a self) -> Option<&'a [T]> {
        let ptr = unsafe { (*self.0.as_ptr()).data };
        if ptr.is_null() {
            event!(Level::ERROR, "variable data pointer is null");
            return None;
        }

        let len = self.size() / size_of::<T>();
        Some(unsafe { slice::from_raw_parts(ptr as *const T, len) })
    }

    fn size(&self) -> usize {
        unsafe { Mat_VarGetSize(self.0.as_ptr()) }
    }
}

impl Drop for Var {
    fn drop(&mut self) {
        unsafe { Mat_VarFree(self.0.as_ptr()) }
    }
}

#[cfg(unix)]
fn path_to_cstring(path: impl AsRef<Path>) -> Result<CString, NulError> {
    use std::os::unix::ffi::OsStrExt;
    CString::new(path.as_ref().as_os_str().as_bytes())
}

/// A variable value.
#[derive(Debug)]
pub enum Value<'a> {
    /// Single-precision float `f32`.
    Single(&'a [f32]),
    /// Double-precision float `f64`.
    Double(&'a [f64]),
    /// Unknown.
    Unknown,
}

impl<'a> Value<'a> {
    pub fn as_single(&self) -> Option<&[f32]> {
        match self {
            Value::Single(v) => Some(v),
            _ => None,
        }
    }
}

/// File access mode.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Access {
    /// Read only.
    Read,
    /// Read-write.
    ReadWrite,
}

impl From<Access> for mat_acc {
    fn from(acc: Access) -> mat_acc {
        match acc {
            Access::Read => mat_acc_MAT_ACC_RDONLY,
            Access::ReadWrite => mat_acc_MAT_ACC_RDWR,
        }
    }
}

/// Errors that can happen when working with MATLAB MAT files.
#[derive(Debug, Error)]
pub enum Error {
    /// The filename contains a NUL byte.
    #[error("NUL byte in file path found")]
    NulInPath(#[from] NulError),
    /// Opening the MATLAB MAT file failed.
    #[error("Failed to open file")]
    OpenFailed,
    /// Variable not found.
    #[error("Variable not found")]
    NotFound,
}
