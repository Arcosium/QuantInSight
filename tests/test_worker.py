import contextlib
import json
import sqlite3
import unittest
from unittest.mock import patch
from autofolio.worker import completed_genomes,process_is_job
from autofolio.genetics import DEFAULT


class WorkerTests(unittest.TestCase):
    def test_atomic_completed_join_during_catalogue_growth(self):
        db=sqlite3.connect(':memory:');db.row_factory=sqlite3.Row
        db.executescript('CREATE TABLE strategies(payload TEXT,source TEXT);CREATE TABLE jobs(id TEXT,result TEXT,status TEXT);')
        db.execute('INSERT INTO strategies VALUES (?,?)',(json.dumps(dict(id='strategy',genome=DEFAULT)), '/book'))
        db.execute('INSERT INTO strategies VALUES (?,?)',(json.dumps(dict(id='newly-indexed',genome=None)), '/other'))
        db.execute('INSERT INTO jobs VALUES (?,?,?)',('job','/book','done'))
        @contextlib.contextmanager
        def connection():yield db
        with patch('autofolio.worker.connect',connection):
            parents=completed_genomes()
        self.assertEqual(len(parents),1)
        self.assertEqual(parents[0]['job_id'],'job')
        self.assertEqual(parents[0]['genome'],DEFAULT)
        db.close()

    def test_pid_reuse_is_not_an_experiment(self):
        import os
        self.assertFalse(process_is_job(os.getpid(),'some-job'))
        self.assertFalse(process_is_job(999999999,'some-job'))


if __name__=='__main__':unittest.main()
