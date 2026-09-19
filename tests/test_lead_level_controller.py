import unittest
from core.lead_level_controller import LeadLevelController

class TestLeadLevelControllerV4(unittest.TestCase):
    def clean(self,c,used,err):
        return c.record_outcome(center_error_ms=err,actual_used_delay_ms=used,outcome='GREAT',
            plateau_found=True,trigger_mode='SCHEDULED',scheduler_jitter_ms=0.2,frame_age_ms=5.0,
            detector_fallback=False,compensation_regime='CONTINUOUS_MEASURED_SPEED',
            fit_sample_count=8,fit_residual_mad_deg=1.0,fit_spread_deg_s=8.0)
    def test_ideal_lead_is_controller_invariant(self):
        c=LeadLevelController(60)
        vals=[self.clean(c,u,e)['ideal_lead_ms'] for u,e in [(60,45),(90,15),(105,0)]]
        self.assertEqual(vals,[105.0,105.0,105.0])
    def test_cold_start_snaps_after_three_consistent_samples(self):
        c=LeadLevelController(60)
        self.clean(c,60,45); self.clean(c,60,44)
        self.assertEqual(c.current_lead_ms,60)
        r=self.clean(c,60,46)
        self.assertTrue(r['updated']); self.assertEqual(r['update_reason'],'COLD_MEDIAN')
        self.assertAlmostEqual(c.current_lead_ms,105.0)
    def test_warm_small_noise_does_not_chase(self):
        c=LeadLevelController(60)
        for e in [45,44,46]: self.clean(c,60,e)
        base=c.current_lead_ms
        for ideal in [108,101,110,99,106,104]: self.clean(c,base,ideal-base)
        self.assertAlmostEqual(c.current_lead_ms,base)
    def test_dirty_sample_cannot_train(self):
        c=LeadLevelController(60)
        r=c.record_outcome(center_error_ms=40,actual_used_delay_ms=60,outcome='GOOD',plateau_found=True,
            trigger_mode='IMMEDIATE',scheduler_jitter_ms=.1,frame_age_ms=2,detector_fallback=False,
            compensation_regime='CONTINUOUS_MEASURED_SPEED',fit_sample_count=8,
            fit_residual_mad_deg=1,fit_spread_deg_s=5)
        self.assertFalse(r['accepted']); self.assertEqual(len(c.samples),0)
    def test_clear_level_shift_moves_only_after_median_changes(self):
        c=LeadLevelController(60)
        for e in [45,44,46]: self.clean(c,60,e)
        self.assertAlmostEqual(c.current_lead_ms,105)
        # Fill most of rolling window with a new stable ~80ms level.
        for ideal in [80,81,79,80,82,78]:
            self.clean(c,c.current_lead_ms,ideal-c.current_lead_ms)
        self.assertLess(c.current_lead_ms,90)

if __name__=='__main__': unittest.main()
