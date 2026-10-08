import secrets
TEST_PASSWORD = secrets.token_urlsafe(24)

import unittest
import tempfile
import os
from autofolio import auth
from unittest.mock import patch
from fastapi.testclient import TestClient
from autofolio.app import app
from autofolio.period import window,accepts,expected_dates
from datetime import date
from autofolio.evaluation import PROTOCOL,periods


def split_summary(identity,span,os_return,whole_return=99,ros_return=99):
    days=expected_dates('crypto',*span);os=periods(*span)['os']
    return dict(id=identity,title=identity,family='test',market='crypto',owner_id=1,
                cohort='cohort-'+span[1],evaluation_window=list(span),months=36,
                start=days[0],end=days[-1],sessions=len(days),phase_count=1,
                net_return=whole_return,negative_months=0,mdd=0,evaluation_protocol=PROTOCOL,
                genome={'engine':'learned_v1'},cases=[],
                performance={'os':dict(net_return=os_return,negative_months=1,mdd=-.1,sharpe=1,
                              months=9,start=os['start'],end=os['end'],sessions=len(expected_dates('crypto',os['start'],os['end']))),
                             'ros':dict(net_return=ros_return)})

class SiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.env=patch.dict(os.environ,{'QUANTINSIGHT_AUTH_DIR':self.tmp.name});self.env.start();self.addCleanup(self.env.stop)
        auth.bootstrap_admin('testowner',TEST_PASSWORD)
        self.client=TestClient(app)
        self.client.post('/api/auth/login',headers={'X-Requested-With':'QuantInSight'},json={'username':'testowner','password':TEST_PASSWORD})

    def test_fixed_calendar_and_short_history_rejected(self):
        self.assertEqual(window(date(2026,10,8)),('20231008','20261007'))
        self.assertEqual(window(date(2024,3,1)),('20210301','20240229'))
        start,end=window()
        days=expected_dates('kr',start,end)
        self.assertTrue(accepts(dict(months=36,start=days[0],end=days[-1])))
        self.assertFalse(accepts(dict(months=36,start=days[0],end=end[:6]+'23')))
        self.assertFalse(accepts(dict(months=33,start=start,end=end)))
        self.assertFalse(accepts(dict(months=36,start='20220101',end='20241231')))
        with patch('autofolio.app.rows',return_value=[dict(months=33,start=start,end=end)]):
            self.assertEqual(self.client.get('/api/leaderboard').json()['total'],0)

    def test_mutations_require_same_origin_header(self):
        body=dict(cpu_cores=4,memory_gb=8,parallel=2)
        self.assertEqual(self.client.post('/api/resources',json=body).status_code,403)
        self.assertEqual(self.client.post('/api/resources',json=body,headers={'X-Requested-With':'QuantInSight','Origin':'https://evil.invalid'}).status_code,403)
        self.assertEqual(self.client.post('/api/resources',json=dict(body,cpu_cores=0),headers={'X-Requested-With':'QuantInSight'}).status_code,422)
        self.assertEqual(self.client.post('/api/seed',json={'strategy':'test strategy'}).status_code,403)

    def test_failed_resource_application_is_not_reported_as_saved(self):
        with patch('autofolio.controls.subprocess.run',side_effect=OSError),patch('autofolio.controls.set_setting') as save:
            response=self.client.post('/api/resources',json=dict(cpu_cores=1,memory_gb=2,parallel=1),headers={'X-Requested-With':'QuantInSight'})
            self.assertEqual(response.status_code,503);save.assert_not_called()

    def test_routes_and_retired_archive(self):
        self.assertEqual(self.client.get('/api/archive').status_code,404)
        self.assertEqual(self.client.get('/api/strategy/unknown?months=33').status_code,422)
        self.assertEqual(self.client.get('/api/accounts/unknown').status_code,404)
        self.assertEqual(self.client.get('/api/logs?after=-1').status_code,422)
        self.assertEqual(self.client.get('/static/../../vault/secrets_api_keys.txt').status_code,404)
        self.assertIn("frame-ancestors 'none'",self.client.get('/').headers['Content-Security-Policy'])
        self.assertEqual(self.client.get('/api/datasets').status_code,200)

    def test_leaderboard_uses_os_only_and_defaults_to_latest_cohort(self):
        from datetime import datetime
        span=window();older=window(datetime.strptime(span[1],'%Y%m%d').date())
        entries=[split_summary('a',span,.1,100,100),split_summary('b',span,.2,-.9,-.9)]
        entries += [split_summary('older'+str(i),older,.9) for i in range(3)]
        legacy=split_summary('legacy',span,100);legacy.pop('evaluation_protocol');entries.append(legacy)
        with patch('autofolio.app.rows',return_value=entries),patch('autofolio.research.protocol',return_value={}),patch('autofolio.map_references.references',return_value={'points':[],'assignments':{}}):
            response=self.client.get('/api/leaderboard?market=crypto')
            self.assertEqual(response.status_code,200)
            payload=response.json()
            self.assertEqual(payload['cohort'],'cohort-'+span[1])
            self.assertEqual(payload['frontier'],['b'])
            self.assertEqual(payload['total'],5)
            self.assertEqual(payload['strategies'][0]['net_return'],.2)
            self.assertEqual(payload['strategies'][0]['months'],9)
            self.assertEqual(payload['selection_scope'],'os')
            historical=self.client.get('/api/leaderboard',params={'market':'crypto','cohort':'cohort-'+older[1]}).json()
            self.assertEqual(len(historical['strategies']),3)

    def test_detail_and_trades_default_to_os_and_keep_all_three_segments(self):
        from datetime import datetime
        span=window(datetime.strptime(window()[1],'%Y%m%d').date())
        summary=split_summary('a'*20,span,.2);days=expected_dates('crypto',*span)
        daily=[dict(date=day,nav=1000*(1.001**(i+1)),cash=0) for i,day in enumerate(days)]
        splits=periods(*span)
        trades=[dict(date=v['start'],code='BTC',side='buy',qty=1,price=1,fee=0) for v in splits.values()]
        case=dict(initial_cash=1000,daily=daily,trades=trades)
        with patch('autofolio.app.strategy_case',return_value=(summary,case)):
            payload=self.client.get('/api/strategy/'+summary['id']).json()
            self.assertEqual(payload['scope'],'os')
            self.assertEqual(payload['metrics']['months'],9)
            self.assertEqual(payload['daily'][0]['date'],splits['os']['start'])
            self.assertEqual(set(payload['performance']),{'is','os','ros'})
            ros=self.client.get('/api/strategy/'+summary['id'],params={'scope':'ros'}).json()
            self.assertEqual(ros['metrics']['months'],3)
            self.assertEqual(ros['daily'][0]['date'],splits['ros']['start'])
            fills=self.client.get('/api/strategy/'+summary['id']+'/trades').json()
            self.assertEqual(fills['total'],1)
            self.assertEqual(fills['trades'][0]['date'],splits['os']['start'])
            whole=self.client.get('/api/strategy/'+summary['id'],params={'scope':'all'}).json()
            self.assertEqual(len(whole['daily']),len(days))
            self.assertEqual(whole['metrics']['months'],36)

if __name__=='__main__':unittest.main()
