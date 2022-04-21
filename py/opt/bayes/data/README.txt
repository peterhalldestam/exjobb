Bayesian Optimization data:


Old data (outdated objective function)

expDecay1_old.json  :   Finished 10+300 runs of ExponentialDecaySimulation
                        Input params:
                            - Injected nD in [1e18, 2e22] (m^-3)
                            - Injected nNe in [1e16, 1e20] (m^-3)
                        Sampled optimum f = 0.45464663122575677 @
                            nD = 1.0338111372611786e22,
                            nNe = 1.6596825608895254e17
                        Note:
                            Posterior mean is constant!

expDecay2_old.json  :   Finished 1+5 runs of ExponentialDecaySimulation
                        Input params:
                            - Injected nD in [1e18, 2e22] (m^-3)
                            - Injected nNe in [1e16, 1e20] (m^-3)
                        Sampled optimum f = 0.48421669704079284 @
                            nD = 1.0338111372611786e22
                            nNe = 1.6596825608895254e17
                        Note:
                            Posterior mean is constant!


New data (updated objective function, optimizes over base-10 logarithmic input space)

expDecay3.json      :   Finished 10+300 runs of ExponentialDecaySimulation
                        Input params:
                            - Injected log_nD in [19, 22.3] (m^-3, log10)
                            - Injected log_nNe in [15, 19] (m^-3, log10)
                        Sampled optimum f = 0.013741051718203615 @
                            log_nD = 22.097658120698522,
                            log_nNe = 16.681081240594324
                        Note:
                            With log10, the posterior mean agrees much more better with datapoints.
