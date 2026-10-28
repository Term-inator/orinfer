//! Preallocated event pairs join one auxiliary stream into the owning graph.
use super::super::{
    api::Handle,
    check,
    runtime::{Context, Session},
};
use crate::artifact::Result;
use std::{any::Any, ptr, rc::Rc};

pub(in crate::backend::ascend) struct Branches {
    context: Rc<Context>,
    streams: Vec<Handle>,
    depths: Vec<usize>,
    events: Vec<(Handle, Handle)>,
    keepalive: Option<Box<dyn Any>>,
}

impl Branches {
    pub(in crate::backend::ascend) fn new(
        session: &Session,
        depths: &[usize],
        keepalive: Box<dyn Any>,
    ) -> Result<Rc<Self>> {
        if !(1..=256).contains(&depths.len()) || depths.iter().any(|&depth| depth >= 8) {
            return Err("ACL Graph requires 1..256 independent branch pairs".into());
        }
        let _current = session.context.enter()?;
        let mut branches = Self {
            context: Rc::clone(&session.context),
            streams: Vec::new(),
            depths: depths.to_vec(),
            events: Vec::with_capacity(depths.len()),
            keepalive: Some(keepalive),
        };
        // SAFETY: Handles are created in the retained owning context. All
        // allocations precede capture; Drop rolls back partial construction.
        unsafe {
            for _ in 0..=depths.iter().copied().max().unwrap() {
                let mut stream = ptr::null_mut();
                check(
                    (branches.context.api.create_stream)(&mut stream),
                    "create ACL Graph branch stream",
                )?;
                branches.streams.push(stream);
            }
            for _ in depths {
                branches.events.push((ptr::null_mut(), ptr::null_mut()));
                let pair = branches.events.last_mut().unwrap();
                // Captured cross-stream dependencies require Ex events with
                // ACL_EVENT_SYNC=1. Each fork has its own event generation.
                check(
                    (branches.context.api.create_capture_event)(&mut pair.0, 1),
                    "create ACL Graph fork event",
                )?;
                check(
                    (branches.context.api.create_capture_event)(&mut pair.1, 1),
                    "create ACL Graph join event",
                )?;
            }
        }
        Ok(Rc::new(branches))
    }

    pub(in crate::backend::ascend) fn fork(&self, index: usize, main: Handle) -> Result<Handle> {
        let pair = self
            .events
            .get(index)
            .ok_or("ACL fork index outside branch table")?;
        let stream = self.streams[self.depths[index]];
        if main == stream
            || (main != self.context.compute_stream() && !self.streams.contains(&main))
        {
            return Err("ACL fork requires a distinct parent stream in the owning context".into());
        }
        // SAFETY: Both streams and the unique event pair belong to this context.
        // The graph/outer completion retains this object and every operand.
        unsafe {
            check(
                (self.context.api.record_event)(pair.0, main),
                "record ACL fork",
            )?;
            check(
                (self.context.api.wait_event)(stream, pair.0, self.context.timeout_ms()),
                "wait for ACL fork",
            )?;
        }
        Ok(stream)
    }

    pub(in crate::backend::ascend) fn switch(&self, index: usize) -> Result<()> {
        let pair = self
            .events
            .get(index)
            .ok_or("ACL switch index outside branch table")?;
        // SAFETY: The side branch is queued before this event on the same stream.
        check(
            unsafe { (self.context.api.record_event)(pair.1, self.streams[self.depths[index]]) },
            "record ACL branch completion",
        )
    }

    pub(in crate::backend::ascend) fn join(&self, index: usize, main: Handle) -> Result<()> {
        let pair = self
            .events
            .get(index)
            .ok_or("ACL join index outside branch table")?;
        // SAFETY: The matching switch has recorded this event. Joining it makes
        // a later completion on main cover all work on the auxiliary stream.
        check(
            unsafe { (self.context.api.wait_event)(main, pair.1, self.context.timeout_ms()) },
            "join ACL branch",
        )
    }
}

impl Drop for Branches {
    fn drop(&mut self) {
        let Ok(_current) = self.context.enter() else {
            std::mem::forget(self.keepalive.take());
            std::mem::forget(Rc::clone(&self.context));
            return;
        };
        // Also fence partial warmup/capture failures before operands or events
        // can disappear. A failed fence quarantines native resources together.
        // SAFETY: Every stream is owned by this retained, current context.
        if self.streams.iter().any(|&stream| unsafe {
            (self.context.api.sync_stream)(stream, self.context.timeout_ms()) != 0
        }) {
            self.context.retire();
            std::mem::forget(self.keepalive.take());
            std::mem::forget(Rc::clone(&self.context));
            return;
        }
        // SAFETY: No side work remains; all successful allocations are owned.
        unsafe {
            for &(start, done) in &self.events {
                if !start.is_null() {
                    (self.context.api.destroy_event)(start);
                }
                if !done.is_null() {
                    (self.context.api.destroy_event)(done);
                }
            }
            for &stream in &self.streams {
                (self.context.api.destroy_stream)(stream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;

    #[test]
    fn branch_resources_roll_back_partial_event_creation() {
        let mut fixture = Fixture::new();
        {
            let session = fixture.session();
            fixture.failure(13);
            assert!(Branches::new(&session, &[0, 1], Box::new(())).is_err());
            fixture.failure(0);
            assert_eq!(fixture.counter(2), 0);
            let branches = Branches::new(&session, &[0, 1], Box::new(())).unwrap();
            assert_eq!(fixture.counter(2), 4);
            assert!(branches.fork(5, session.compute_stream()).is_err());
            drop(branches);
            assert_eq!(fixture.counter(2), 0);
        }
        fixture.assert_clean();
    }
}
