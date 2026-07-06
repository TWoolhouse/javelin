use tokio::sync::mpsc;

pub trait Actionable<A> {
    fn action(&mut self, action: A) -> Option<A>;
}

impl<T, A> Actionable<A> for Option<T>
where
    T: Actionable<A>,
{
    fn action(&mut self, action: A) -> Option<A> {
        match self {
            Some(on) => on.action(action),
            None => Some(action),
        }
    }
}

pub struct Packet<A> {
    pub action: A,
    pub tx: mpsc::UnboundedSender<A>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Tick,
    Quit,
}
