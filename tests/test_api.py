import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from autofolio.app import app
from autofolio.period import window,accepts
from datetime import date

class SiteTests(unittest.TestCase):
    def setUp(self):self.client=TestClient(app)

    def test_fixed_calendar_and_short_history_rejected(self):
        self.assertEqual(window(date(2026,10,8)),('20231001','20260930'))
        self.assertEqual(window(date(2024,3,1)),('20210301','20240229'))
        start,end=window()
        self.assertTrue(accepts(dict(months=36,start=start,end=end)))
        self.assertFalse(accepts(dict(months=33,start=start,end=end)))
        self.assertFalse(accepts(dict(months=36,start='20220101',end='20241231')))
        with patch('autofolio.app.rows',return_value=[dict(months=33,start=start,end=end)]):
            self.assertEqual(self.client.get('/api/leaderboard').json()['total'],0)

    def test_mutations_require_same_origin_header(self):
        body=dict(cpu_cores=4,memory_gb=8,parallel=2)
        self.assertEqual(self.client.post('/api/resources',json=body).status_code,403)
        self.assertEqual(self.client.post('/api/resources',json=body,headers={'X-Requested-With':'QuantInSight','Origin':'https://evil.invalid'}).status_code,403)
        self.assertEqual(self.client.post('/api/resources',json=dict(body,cpu_cores=0)).status_code,422)
        self.assertEqual(self.client.post('/api/seed',json={'strategy':'test strategy'}).status_code,403)

    def test_failed_resource_application_is_not_reported_as_saved(self):
        with patch('autofolio.controls.subprocess.run',side_effect=OSError),patch('autofolio.controls.set_setting') as save:
            response=self.client.post('/api/resources',json=dict(cpu_cores=1,memory_gb=2,parallel=1),headers={'X-Requested-With':'QuantInSight'})
            self.assertEqual(response.status_code,503);save.assert_not_called()

    def test_routes_and_retired_archive(self):
        self.assertEqual(self.client.get('/api/archive').status_code,404)
        self.assertEqual(self.client.get('/api/accounts/unknown').status_code,404)
        self.assertEqual(self.client.get('/api/logs?after=-1').status_code,422)
        self.assertEqual(self.client.get('/static/../../vault/secrets_api_keys.txt').status_code,404)
        self.assertIn("frame-ancestors 'none'",self.client.get('/').headers['Content-Security-Policy'])
        self.assertEqual(self.client.get('/api/datasets').status_code,200)

if __name__=='__main__':unittest.main()
