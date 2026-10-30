#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to arm_control__msg__PosCmd

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PosCmd::default())
  }
}

impl rosidl_runtime_rs::Message for PosCmd {
  type RmwMsg = super::msg::rmw::PosCmd;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        z: msg.z,
        roll: msg.roll,
        pitch: msg.pitch,
        yaw: msg.yaw,
        gripper: msg.gripper,
        quater_x: msg.quater_x,
        quater_y: msg.quater_y,
        quater_z: msg.quater_z,
        quater_w: msg.quater_w,
        chx: msg.chx,
        chy: msg.chy,
        chz: msg.chz,
        vel_l: msg.vel_l,
        vel_r: msg.vel_r,
        height: msg.height,
        head_pit: msg.head_pit,
        head_yaw: msg.head_yaw,
        temp_float_data: msg.temp_float_data,
        temp_int_data: msg.temp_int_data,
        mode1: msg.mode1,
        mode2: msg.mode2,
        time_count: msg.time_count,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      roll: msg.roll,
      pitch: msg.pitch,
      yaw: msg.yaw,
      gripper: msg.gripper,
      quater_x: msg.quater_x,
      quater_y: msg.quater_y,
      quater_z: msg.quater_z,
      quater_w: msg.quater_w,
      chx: msg.chx,
      chy: msg.chy,
      chz: msg.chz,
      vel_l: msg.vel_l,
      vel_r: msg.vel_r,
      height: msg.height,
      head_pit: msg.head_pit,
      head_yaw: msg.head_yaw,
        temp_float_data: msg.temp_float_data,
        temp_int_data: msg.temp_int_data,
      mode1: msg.mode1,
      mode2: msg.mode2,
      time_count: msg.time_count,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      roll: msg.roll,
      pitch: msg.pitch,
      yaw: msg.yaw,
      gripper: msg.gripper,
      quater_x: msg.quater_x,
      quater_y: msg.quater_y,
      quater_z: msg.quater_z,
      quater_w: msg.quater_w,
      chx: msg.chx,
      chy: msg.chy,
      chz: msg.chz,
      vel_l: msg.vel_l,
      vel_r: msg.vel_r,
      height: msg.height,
      head_pit: msg.head_pit,
      head_yaw: msg.head_yaw,
      temp_float_data: msg.temp_float_data,
      temp_int_data: msg.temp_int_data,
      mode1: msg.mode1,
      mode2: msg.mode2,
      time_count: msg.time_count,
    }
  }
}


// Corresponds to arm_control__msg__JointControl

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointControl::default())
  }
}

impl rosidl_runtime_rs::Message for JointControl {
  type RmwMsg = super::msg::rmw::JointControl;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        joint_pos: msg.joint_pos,
        joint_vel: msg.joint_vel,
        joint_cur: msg.joint_cur,
        mode: msg.mode,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        joint_pos: msg.joint_pos,
        joint_vel: msg.joint_vel,
        joint_cur: msg.joint_cur,
      mode: msg.mode,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      joint_pos: msg.joint_pos,
      joint_vel: msg.joint_vel,
      joint_cur: msg.joint_cur,
      mode: msg.mode,
    }
  }
}


// Corresponds to arm_control__msg__ArxImu

// This struct is not documented.
#[allow(missing_docs)]

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ArxImu {

    // This member is not documented.
    #[allow(missing_docs)]
    pub stamp: builtin_interfaces::msg::Time,


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
    pub angular_velocity: geometry_msgs::msg::Vector3,

}



impl Default for ArxImu {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::ArxImu::default())
  }
}

impl rosidl_runtime_rs::Message for ArxImu {
  type RmwMsg = super::msg::rmw::ArxImu;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        stamp: builtin_interfaces::msg::Time::into_rmw_message(std::borrow::Cow::Owned(msg.stamp)).into_owned(),
        roll: msg.roll,
        pitch: msg.pitch,
        yaw: msg.yaw,
        angular_velocity: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.angular_velocity)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        stamp: builtin_interfaces::msg::Time::into_rmw_message(std::borrow::Cow::Borrowed(&msg.stamp)).into_owned(),
      roll: msg.roll,
      pitch: msg.pitch,
      yaw: msg.yaw,
        angular_velocity: geometry_msgs::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.angular_velocity)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      stamp: builtin_interfaces::msg::Time::from_rmw_message(msg.stamp),
      roll: msg.roll,
      pitch: msg.pitch,
      yaw: msg.yaw,
      angular_velocity: geometry_msgs::msg::Vector3::from_rmw_message(msg.angular_velocity),
    }
  }
}


