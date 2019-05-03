import numpy as np
import json
class CDF:
    def __init__(self, x, F):
        s = sorted(zip(x, F)) # Make sure that x is sorted in ascending order, since combine assumes that
        self.x = [t[0] for t in s]
        self.F = [t[1] for t in s]
    
    def combine(cdfs):
        x_unique = set()
        for cdf in cdfs:
            x_unique.update(cdf.x)

        x_combined = [t for t in sorted(x_unique)]

        weights = [1.0 / len(cdfs)] * len(cdfs)
        pos = [0] * len(cdfs)
        combined_cdf = {'x': [], 'F': []}

        for x in x_combined:
            weighted_F = 0
            for j in range(len(cdfs)):
                w = weights[j]
                x_j = cdfs[j].x
                F_j = cdfs[j].F
                p = pos[j]
                while p + 1 < len(x_j) and x_j[p+1] < x:
                    p += 1
                pos[j] = p

                if x < x_j[p]:
                    # x is lower than the lowest value in CDF[j] so we take F=0
                    F = F_j[p]
                elif p == len(x_j) - 1:
                    # x greater or equal than the largest value in CDF[j], so we take F=1
                    F = F_j[p]
                else:
                    # x is between x_j[p] and x_j[p+1], so we interpolate F
                    y_interp = np.interp(x=[x], xp=[x_j[p], x_j[p+1]], fp=[F_j[p], F_j[p+1]])
                    F = y_interp[0]

                weighted_F += w * F
            combined_cdf['x'].append(x)
            combined_cdf['F'].append(weighted_F)
        return CDF(x=combined_cdf['x'], F=combined_cdf['F'])
    
    def rvs(self, n):
        res = []
        for i in range(n):
            prev = None
            p = np.random.random()
            for x, F in zip(self.x, self.F):
                if F >= p:
                    if prev is None:
                        res.append(x)
                    else:
                        y_target = p
                        x_interp = np.interp(x=[y_target], xp=[prev['F'], F], fp=[prev['x'], x])[0]
                        res.append(x_interp)
                    break
                else:
                    prev = {'x': x, 'F': F}
                    
        if len(res) == 0:
            return self.x[-1]
        if n > 1:
            return np.array(res)
        else:
            return res[0]
        
    def get_raw_data(self):
        return {'xs': self.x, 'ys': self.F}
        
    def to_json(self):
        return json.dumps({'xs': self.x, 'ys': self.F})
                

