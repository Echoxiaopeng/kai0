#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};


#[link(name = "arm_control__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__PosCmd() -> *const std::ffi::c_void;
}

#[link(name = "arm_control__rosidl_generator_c")]
extern "C" {
    fn arm_control__msg__PosCmd__init(msg: *mut PosCmd) -> bool;
    fn arm_control__msg__PosCmd__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<PosCmd>, size: usize) -> bool;
    fn arm_control__msg__PosCmd__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<PosCmd>);
    fn arm_control__msg__PosCmd__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<PosCmd>, out_seq: *mut rosidl_runtime_rs::Sequence<PosCmd>) -> bool;
}

// Corresponds to arm_control__msg__PosCmd
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PosCmd {

    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub z: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub roll: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pitch: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub yaw: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub gripper: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub quater_x: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub quater_y: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub quater_z: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub quater_w: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub chx: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub chy: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub chz: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub vel_l: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub vel_r: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub height: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub head_pit: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub head_yaw: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub temp_float_data: [f64; 6],


    // This member is not documented.
    #[allow(missing_docs)]
    pub temp_int_data: [i32; 6],


    // This member is not documented.
    #[allow(missing_docs)]
    pub mode1: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub mode2: i32,


    // This member is not documented.
    #[allow(missing_docs)]
    pub time_count: i32,

}



impl Default for PosCmd {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !arm_control__msg__PosCmd__init(&mut msg as *mut _) {
        panic!("Call to arm_control__msg__PosCmd__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for PosCmd {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__PosCmd__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__PosCmd__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__PosCmd__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for PosCmd {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for PosCmd where Self: Sized {
  const TYPE_NAME: &'static str = "arm_control/msg/PosCmd";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__PosCmd() }
  }
}


#[link(name = "arm_control__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__JointControl() -> *const std::ffi::c_void;
}

#[link(name = "arm_control__rosidl_generator_c")]
extern "C" {
    fn arm_control__msg__JointControl__init(msg: *mut JointControl) -> bool;
    fn arm_control__msg__JointControl__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<JointControl>, size: usize) -> bool;
    fn arm_control__msg__JointControl__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<JointControl>);
    fn arm_control__msg__JointControl__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<JointControl>, out_seq: *mut rosidl_runtime_rs::Sequence<JointControl>) -> bool;
}

// Corresponds to arm_control__msg__JointControl
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointControl {

    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_pos: [f32; 8],


    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_vel: [f32; 8],


    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_cur: [f32; 8],


    // This member is not documented.
    #[allow(missing_docs)]
    pub mode: i32,

}



impl Default for JointControl {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !arm_control__msg__JointControl__init(&mut msg as *mut _) {
        panic!("Call to arm_control__msg__JointControl__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for JointControl {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__JointControl__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__JointControl__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__JointControl__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for JointControl {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for JointControl where Self: Sized {
  const TYPE_NAME: &'static str = "arm_control/msg/JointControl";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__JointControl() }
  }
}


#[link(name = "arm_control__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__ArxImu() -> *const std::ffi::c_void;
}

#[link(name = "arm_control__rosidl_generator_c")]
extern "C" {
    fn arm_control__msg__ArxImu__init(msg: *mut ArxImu) -> bool;
    fn arm_control__msg__ArxImu__Sequence__init(seq: *mut rosidl_runtime_rs::Sequence<ArxImu>, size: usize) -> bool;
    fn arm_control__msg__ArxImu__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<ArxImu>);
    fn arm_control__msg__ArxImu__Sequence__copy(in_seq: &rosidl_runtime_rs::Sequence<ArxImu>, out_seq: *mut rosidl_runtime_rs::Sequence<ArxImu>) -> bool;
}

// Corresponds to arm_control__msg__ArxImu
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]


// This struct is not documented.
#[allow(missing_docs)]

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ArxImu {

    // This member is not documented.
    #[allow(missing_docs)]
    pub stamp: builtin_interfaces::msg::rmw::Time,


    // This member is not documented.
    #[allow(missing_docs)]
    pub roll: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pitch: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub yaw: f64,


    // This member is not documented.
    #[allow(missing_docs)]
    pub angular_velocity: geometry_msgs::msg::rmw::Vector3,

}



impl Default for ArxImu {
  fn default() -> Self {
    unsafe {
      let mut msg = std::mem::zeroed();
      if !arm_control__msg__ArxImu__init(&mut msg as *mut _) {
        panic!("Call to arm_control__msg__ArxImu__init() failed");
      }
      msg
    }
  }
}

impl rosidl_runtime_rs::SequenceAlloc for ArxImu {
  fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__ArxImu__Sequence__init(seq as *mut _, size) }
  }
  fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__ArxImu__Sequence__fini(seq as *mut _) }
  }
  fn sequence_copy(in_seq: &rosidl_runtime_rs::Sequence<Self>, out_seq: &mut rosidl_runtime_rs::Sequence<Self>) -> bool {
    // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
    unsafe { arm_control__msg__ArxImu__Sequence__copy(in_seq, out_seq as *mut _) }
  }
}

impl rosidl_runtime_rs::Message for ArxImu {
  type RmwMsg = Self;
  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> { msg_cow }
  fn from_rmw_message(msg: Self::RmwMsg) -> Self { msg }
}

impl rosidl_runtime_rs::RmwMessage for ArxImu where Self: Sized {
  const TYPE_NAME: &'static str = "arm_control/msg/ArxImu";
  fn get_type_support() -> *const std::ffi::c_void {
    // SAFETY: No preconditions for this function.
    unsafe { rosidl_typesupport_c__get_message_type_support_handle__arm_control__msg__ArxImu() }
  }
}


