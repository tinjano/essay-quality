import polars as pl
import functools as ft

class MFDFA:
    def __init__(self, df, target):
        if 'event_id' in df.columns:
            df.drop_in_place('event_id')
        
        q = df.lazy().with_columns(
            (pl.col(target).is_nan().not_()).cum_sum().over('id').alias('event_id')
        )
        
        frame = q.collect()
        
        q = frame.lazy().select(
            pl.col('id'),
            pl.col('event_id'),
            pl.col(target).cum_sum().over('id'),
        )
        
        self.base_frame = q.collect()
        self.base_target = target
        self.s = set()
    
    @ft.cache
    def _add_s(self, s, target=None, source='event_id'):
        target = self.base_target if target is None else target
        self.s.add(s)
        
        q = self.base_frame.lazy().with_columns(
            (pl.col('event_id') / s).over('id').ceil().alias('fw_bracket'),
            ((pl.col('event_id').max() + 1 - pl.col('event_id')) / s).over('id').ceil().alias('bk_bracket')
        )
        
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col(source) - (pl.col(source).mean().over('id', 'fw_bracket'))).alias('fw_dt'),
            (pl.col(target) - (pl.col(target).mean().over('id', 'fw_bracket'))).alias('fw_dx'),
            (pl.col(source) - (pl.col(source).mean().over('id', 'bk_bracket'))).alias('bk_dt'),
            (pl.col(target) - (pl.col(target).mean().over('id', 'bk_bracket'))).alias('bk_dx')
        )
        
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            ((pl.col('fw_dt') * pl.col('fw_dx')).sum().over('id', 'fw_bracket') / (pl.col('fw_dt') ** 2).sum().over('id', 'fw_bracket')).alias('fw_beta'),
            ((pl.col('bk_dt') * pl.col('bk_dx')).sum().over('id', 'bk_bracket') / (pl.col('bk_dt') ** 2).sum().over('id', 'bk_bracket')).alias('bk_beta'),
        )
    
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col('fw_dx') - pl.col('fw_beta') * pl.col('fw_dt')).alias('fw_delta'),
            (pl.col('bk_dx') - pl.col('bk_beta') * pl.col('bk_dt')).alias('bk_delta'),
        )
    
        new_frame = q.collect()
        
        q = new_frame.lazy().with_columns(
            (pl.col('fw_delta') ** 2).mean().over('id', 'fw_bracket').alias('fw_F2'),
            (pl.col('bk_delta') ** 2).mean().over('id', 'bk_bracket').alias('bk_F2'),
        )
    
        new_frame = q.collect()
        
        fw_frame = new_frame.group_by('id', 'fw_bracket').agg(pl.col('fw_F2').drop_nans().first())
        bk_frame = new_frame.group_by('id', 'bk_bracket').agg(pl.col('bk_F2').drop_nans().first())
        
        return fw_frame, bk_frame
    
    def add_s_multi(self, s_iter, target=None, source='event_id'):
        target = self.base_target if target is None else target
        [self._add_s(s, target, source) for s in s_iter]
    
    def add_q(self, q, target=None, source='event_id'):
        target = self.base_target if target is None else target
        ans = self.base_frame.select(pl.col('id')).group_by('id').agg(pl.all().first())
        
        for s in self.s:
            fw_frame, bk_frame = self._add_s(s, target, source)
            
            fw_frame = fw_frame.group_by('id').agg(
                (pl.col('fw_F2') ** (q/2)).mean()
            )
            
            bk_frame = bk_frame.group_by('id').agg(
                (pl.col('bk_F2') ** (q/2)).mean()
            )
            
            frame = fw_frame.join(
                bk_frame,
                on='id'
            )
                    
            frame = frame.select(
                pl.col('id'),
                (((pl.col('fw_F2') + pl.col('bk_F2')) / 2) ** (1/q)).alias(f'F_{q}({s})')
            )
        
            ans = ans.join(
                frame,
                on='id'
            )
            
        return ans
    
    def add_q_multi(self, q_iter, target=None, source='event_id'):
        target = self.base_target if target is None else target
        ans = self.base_frame.select(pl.col('id')).group_by('id').agg(pl.all().first())        
        
        for q in q_iter:
            ans = ans.join(
                self.add_q(q, target, source),
                on='id'
            )
        
        return ans
    
def R2(y, y_hat):
    """ Vectorized R^2 (coefficient of determination) function """
    if (y.ndim < 1 or y.shape != y_hat.shape):
        raise IndexError('`y` and `y_hat` must have the same shape.')
    
    y_mean = y.mean(axis=y.ndim - 1, keepdims=True)
    N = ((y_hat - y_mean) ** 2).sum(axis=y.ndim - 1)
    D = ((y - y_mean) ** 2).sum(axis=y.ndim - 1)
    
    result = N / D
    nan_mask = D == 0
    result[nan_mask] = 1
    
    return result
