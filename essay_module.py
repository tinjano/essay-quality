import re
import pandas as pd
from tqdm.notebook import tqdm; tqdm.pandas()
        
class Extractor:
    
    def __init__(self, *funcs):
        self.funcs = funcs
            
    def process_group(self, group):
        """
        inspired by:
        https://www.kaggle.com/code/kawaiicoderuwu/essay-contructor
        https://www.kaggle.com/code/yuriao/fast-essay-constructor
        """
        
        move_pattern = re.compile(r'Move From \[(\d+), (\d+)\] To \[(\d+), (\d+)\]')

        def generator(_id=None, buffer=''):
            
            for row in group.values:
                index, activity, cursor_pos, text_change =\
                        row[0], row[1], row[2], row[3]
                
                if _id != index:
                    buffer = ''
                    _id = index

                match activity:
                    case 'Paste':
                        cursor_pos -= len(text_change)
                        buffer = buffer[:cursor_pos] + text_change + buffer[cursor_pos:]

                    case 'Replace':
                        old_text, new_text = text_change.split(' => ')
                        cursor_pos -= len(new_text)
                        buffer = buffer[:cursor_pos] + new_text + buffer[cursor_pos + len(old_text):]

                    case 'Remove/Cut':
                        buffer = buffer[:cursor_pos] + buffer[cursor_pos + len(text_change):]

                    case activity if activity.startswith('M'):
                        f1, t1, f2, t2 = map(int, move_pattern.match(activity).groups())

                        if f1 < f2:
                            buffer = buffer[:f1] + buffer[t1:t2] + buffer[f1:t1] + buffer[t2:]
                        elif f1 > f2:
                            buffer = buffer[:f2] + buffer[f1:t1] + buffer[f2:f1] + buffer[t1:]

                    case 'Input':
                        cursor_pos -= len(text_change)
                        buffer = buffer[:cursor_pos] + text_change + buffer[cursor_pos:]

                    case _:
                        pass
                
                yield {func.__name__ : func(buffer) for func in self.funcs}
                    
        return pd.DataFrame(generator())
    
    
    def do_it(self, df):
        new_frame = df.groupby('id')[['id', 'activity', 'cursor_position', 'text_change']].progress_apply(lambda x: self.process_group(x))
        return new_frame
        
        
        
        
                