Added
^^^^^

* Added :meth:`~isaaclab.managers.RewardManager.get_term_statistics` and
  :meth:`~isaaclab.managers.TerminationManager.get_term_statistics`, which report
  cross-environment statistics (mean, standard deviation, min, max, top-k share, and share of
  total for reward terms; activation rate for termination terms) for each active term, instead
  of the single cross-environment mean currently logged. This makes it possible to spot a reward
  term that pays nothing, a term that dominates the total or is earned disproportionately by a
  few environments, or a termination condition that never fires.
