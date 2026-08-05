"""Tests for studio job state migration."""

from satmap_dataset.studio.jobs import Job, JobState, JobStatus, migrate_job, migrate_job_state


def test_migrate_job_state_backfills_progress_fields():
    legacy = JobState(status=JobStatus.RUNNING, message="old")
    migrated = migrate_job_state(legacy)
    assert migrated.progress_current == 0
    assert migrated.progress_total == 0
    assert migrated.progress_label == ""
    assert migrated.logs == []


def test_migrate_job_on_job_instance():
    job = Job(name="index", state=JobState(status=JobStatus.SUCCESS, message="done"))
    migrate_job(job)
    assert job.state.progress_total == 0
    assert job.state.logs == []
