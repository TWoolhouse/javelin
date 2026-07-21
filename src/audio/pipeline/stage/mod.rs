use crate::audio::pipeline::module::{PassSpec, PostStage as PostStageTrait, PostStageBuilder};

pub mod notes;
pub mod smooth;

macro_rules! stages {
    (
		$(#[$meta:meta])*
		$vis:vis enum $name:ident {
			$($field:ident($ty:ty)),*$(,)?
		}
		impl $target:ident;
	) => {
		$(#[$meta])*
		$vis enum $name {
			$($field($ty)),*
		}

		impl $name {
			$vis fn build(&mut self, spec: &PassSpec) -> Result<Box<dyn $target>, ()> {
				match self {
					$(Self::$field(stage) => stage.build(spec).map(|s| Box::new(s) as Box<dyn $target>)),*
				}
			}
		}

		$(impl From<$ty> for $name {
			fn from(value: $ty) -> Self {
				Self::$field(value)
			}
		})*
	};
}

stages! {
#[derive(Debug, Clone, PartialEq)]
pub enum PostStage {
    Notes(notes::Notes),
    Smooth(smooth::Smooth),
}
impl PostStageTrait;
}
