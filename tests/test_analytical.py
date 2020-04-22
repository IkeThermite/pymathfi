import unittest
import analytical.pricing as pricing

class TestPricing(unittest.TestCase):
    def setUp(self):
        self.S0 = 100
        self.T = 2
        self.r = 0.05
        self.sig = 0.2        

    def test_price_put_BS(self):
        strikes = (180, 100, 20)
        target_prices = (63.487654789203376, 6.610521528574566, 
            0.000000001412420)
        for i in range(3):
            self.assertAlmostEqual(pricing.price_put_BS(
                self.S0, strikes[i], self.T, self.r, self.sig), 
                target_prices[i])

    def test_price_call_BS(self):
        strikes = (180, 100, 20)
        target_prices = (0.616919542730657, 16.126779724978633, 
            81.90325164069322)
        for i in range(3):
            self.assertAlmostEqual(pricing.price_call_BS(
                self.S0, strikes[i], self.T, self.r, self.sig), 
                target_prices[i])

    def test_price_put_CEV(self):
        strikes = (60, 100, 140)
        alphas = (0.2, 0.5, 0.9)
        sig_CEV = (8, 2, 0.4)
        target_prices = [
            [0.395102341625255, 6.682237364858608, 29.220025067034527], 
            [0.254691434287102, 6.619770490971426, 29.553505240830873], 
            [0.526050894195097, 9.252524142765573, 32.526199550777335]]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(pricing.price_put_CEV(
                    self.S0, strikes[j], self.T, self.r, sig_CEV[i], alphas[i]),
                    target_prices[i][j])

    def test_price_call_CEV(self):
        strikes = (60, 100, 140)
        alphas = (0.2, 0.5, 0.9)
        sig_CEV = (8, 2, 0.4)
        target_prices = [
            [46.104857259467686, 16.198495561262661, 2.542786542000208], 
            [45.964446352129528, 16.136028687375472, 2.876266715796540], 
            [46.235805812037526, 18.768782339169633, 5.848961025742998]]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(pricing.price_call_CEV(
                    self.S0, strikes[j], self.T, self.r, sig_CEV[i], alphas[i]),
                    target_prices[i][j])
