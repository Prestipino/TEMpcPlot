def Trans2Text(Trans):
    "from transformation matrix to text"
    cells = ['a', 'b', 'c']
    Text = ''
    for row in Trans:
        Fld = ''
        for i in [0, 1, 2]:
            if row[i]:
                if Fld and row[i] > 0.:
                    Fld += '+'
                Fld += '%3.1f' % (row[i]) + cells[i]
        Text += Fld
        Text += ','
        Text = Text.replace('1.0', '').replace('.0', '').replace('0.5', '1/2')
    return Text[:-1]

def getlattSym(Trans):
    Fives = {'ababc': 'abc', 'bcbca': 'cba', 'acacb': 'acb',
             'cabab': 'cab', 'abcab': 'acb'}
    transText = Trans2Text(Trans)
    lattSym = ''
    for fld in transText.split(','):
        if 'a' in fld:
            lattSym += 'a'
        if 'b' in fld:
            lattSym += 'b'
        if 'c' in fld:
            lattSym += 'c'
    if len(lattSym) != 3:
        lattSym = 'abc'
#        lattSym = Fives[lattSym]
    return lattSym


spgbyNum = [None,
            'P 1', 'P -1',                                                 # 1-
            'P 2', 'P 21', 'C 2', 'P m', 'P c', 'C m', 'C c', 'P 2/m', 'P 21/m',
            'C 2/m', 'P 2/c', 'P 21/c', 'C 2/c',                          # 3-1
            'P 2 2 2', 'P 2 2 21', 'P 21 21 2', 'P 21 21 21',
            'C 2 2 21', 'C 2 2 2', 'F 2 2 2', 'I 2 2 2', 'I 21 21 21',
            'P m m 2', 'P m c 21', 'P c c 2', 'P m a 2', 'P c a 21',
            'P n c 2', 'P m n 21', 'P b a 2', 'P n a 21', 'P n n 2',
            'C m m 2', 'C m c 21', 'C c c 2',
            'A m m 2', 'A b m 2', 'A m a 2', 'A b a 2',
            'F m m 2', 'F d d 2', 'I m m 2', 'I b a 2', 'I m a 2',
            'P m m m', 'P n n n', 'P c c m', 'P b a n',
            'P m m a', 'P n n a', 'P m n a', 'P c c a', 'P b a m', 'P c c n',
            'P b c m', 'P n n m', 'P m m n', 'P b c n', 'P b c a', 'P n m a',
            'C m c m', 'C m c a', 'C m m m', 'C c c m', 'C m m a', 'C c c a',
            'F m m m', 'F d d d',
            'I m m m', 'I b a m', 'I b c a', 'I m m a',                  # 16-7
            'P 4', 'P 41', 'P 42', 'P 43',
            'I 4', 'I 41',
            'P -4', 'I -4', 'P 4/m', 'P 42/m', 'P 4/n', 'P 42/n',
            'I 4/m', 'I 41/a',
            'P 4 2 2', 'P 4 21 2', 'P 41 2 2', 'P 41 21 2', 'P 42 2 2',
            'P 42 21 2', 'P 43 2 2', 'P 43 21 2',
            'I 4 2 2', 'I 41 2 2',
            'P 4 m m', 'P 4 b m', 'P 42 c m', 'P 42 n m', 'P 4 c c', 'P 4 n c',
            'P 42 m c', 'P 42 b c',
            'I 4 m m', 'I 4 c m', 'I 41 m d', 'I 41 c d',
            'P -4 2 m', 'P -4 2 c', 'P -4 21 m', 'P -4 21 c', 'P -4 m 2',
            'P -4 c 2', 'P -4 b 2', 'P -4 n 2',
            'I -4 m 2', 'I -4 c 2', 'I -4 2 m', 'I -4 2 d',
            'P 4/m m m', 'P 4/m c c', 'P 4/n b m', 'P 4/n n c', 'P 4/m b m',
            'P 4/m n c', 'P 4/n m m', 'P 4/n c c', 'P 42/m m c', 'P 42/m c m',
            'P 42/n b c', 'P 42/n n m', 'P 42/m b c', 'P 42/m n m', 'P 42/n m c',
            'P 42/n c m',
            'I 4/m m m', 'I 4/m c m', 'I 41/a m d', 'I 41/a c d',
            'P 3', 'P 31', 'P 32', 'R 3', 'P -3', 'R -3',
            'P 3 1 2', 'P 3 2 1', 'P 31 1 2', 'P 31 2 1', 'P 32 1 2', 'P 32 2 1',
            'R 3 2',
            'P 3 m 1', 'P 3 1 m', 'P 3 c 1', 'P 3 1 c',
            'R 3 m', 'R 3 c',
            'P -3 1 m', 'P -3 1 c', 'P -3 m 1', 'P -3 c 1',
            'R -3 m', 'R -3 c',                                       # 75-16
            'P 6', 'P 61',
            'P 65', 'P 62', 'P 64', 'P 63', 'P -6', 'P 6/m', 'P 63/m', 'P 6 2 2',
            'P 61 2 2', 'P 65 2 2', 'P 62 2 2', 'P 64 2 2', 'P 63 2 2', 'P 6 m m',
            'P 6 c c', 'P 63 c m', 'P 63 m c', 'P -6 m 2', 'P -6 c 2', 'P -6 2 m',
            'P -6 2 c', 'P 6/m m m', 'P 6/m c c', 'P 63/m c m', 'P 63/m m c', # 168-19
            'P 2 3', 'F 2 3', 'I 2 3', 'P 21 3', 'I 21 3', 'P m 3', 'P n 3',
            'F m -3', 'F d -3', 'I m -3',
            'P a -3', 'I a -3', 'P 4 3 2', 'P 42 3 2', 'F 4 3 2', 'F 41 3 2',
            'I 4 3 2', 'P 43 3 2', 'P 41 3 2', 'I 41 3 2', 'P -4 3 m',
            'F -4 3 m', 'I -4 3 m', 'P -4 3 n', 'F -4 3 c', 'I -4 3 d',
            'P m -3 m', 'P n -3 n', 'P m -3 n', 'P n -3 m',
            'F m -3 m', 'F m -3 c', 'F d -3 m', 'F d -3 c',
            'I m -3 m', 'I a -3 d', ]                               # 195-23
altSettingOrtho = {}
''' A dictionary of alternate settings for orthorhombic unit cells
'''
altSettingOrtho = {
        'P 2 2 2': {'abc': 'P 2 2 2', 'cab': 'P 2 2 2', 'bca': 'P 2 2 2',
                    'acb': 'P 2 2 2', 'bac': 'P 2 2 2', 'cba': 'P 2 2 2'},
        'P 2 2 21': {'abc': 'P 2 2 21', 'cab': 'P 21 2 2', 'bca': 'P 2 21 2',
                     'acb': 'P 2 21 2', 'bac': 'P 2 2 21', 'cba': 'P 21 2 2'},
        'P 21 21 2': {'abc': 'P 21 21 2', 'cab': 'P 2 21 21',
                      'bca': 'P 21 2 21', 'acb': 'P 21 2 21',
                      'bac': 'P 21 21 2', 'cba': 'P 2 21 21'},
        'P 21 21 21': {'abc': 'P 21 21 21', 'cab': 'P 21 21 21',
                       'bca': 'P 21 21 21', 'acb': 'P 21 21 21',
                       'bac': 'P 21 21 21', 'cba': 'P 21 21 21'},
        'C 2 2 21': {'abc': 'C 2 2 21', 'cab': 'A 21 2 2', 'bca': 'B 2 21 2',
                     'acb': 'B 2 21 2', 'bac': 'C 2 2 21', 'cba': 'A 21 2 2'},
        'C 2 2 2': {'abc': 'C 2 2 2', 'cab': 'A 2 2 2', 'bca': 'B 2 2 2',
                    'acb': 'B 2 2 2', 'bac': 'C 2 2 2', 'cba': 'A 2 2 2'},
        'F 2 2 2': {'abc': 'F 2 2 2', 'cab': 'F 2 2 2', 'bca': 'F 2 2 2',
                    'acb': 'F 2 2 2', 'bac': 'F 2 2 2', 'cba': 'F 2 2 2'},
        'I 2 2 2': {'abc': 'I 2 2 2', 'cab': 'I 2 2 2', 'bca': 'I 2 2 2',
                    'acb': 'I 2 2 2', 'bac': 'I 2 2 2', 'cba': 'I 2 2 2'},
        'I 21 21 21': {'abc': 'I 21 21 21', 'cab': 'I 21 21 21',
                       'bca': 'I 21 21 21', 'acb': 'I 21 21 21',
                       'bac': 'I 21 21 21', 'cba': 'I 21 21 21'},
        'P m m 2': {'abc': 'P m m 2', 'cab': 'P 2 m m', 'bca': 'P m 2 m',
                    'acb': 'P m 2 m', 'bac': 'P m m 2', 'cba': 'P 2 m m'},
        'P m c 21': {'abc': 'P m c 21', 'cab': 'P 21 m a', 'bca': 'P b 21 m',
                     'acb': 'P m 21 b', 'bac': 'P c m 21', 'cba': 'P 21 a m'},
        'P c c 2': {'abc': 'P c c 2', 'cab': 'P 2 a a', 'bca': 'P b 2 b',
                    'acb': 'P b 2 b', 'bac': 'P c c 2', 'cba': 'P 2 a a'},
        'P m a 2': {'abc': 'P m a 2', 'cab': 'P 2 m b', 'bca': 'P c 2 m',
                    'acb': 'P m 2 a', 'bac': 'P b m 2', 'cba': 'P 2 c m'},
        'P c a 21': {'abc': 'P c a 21', 'cab': 'P 21 a b', 'bca': 'P c 21 b',
                     'acb': 'P b 21 a', 'bac': 'P b c 21', 'cba': 'P 21 c a'},
        'P n c 2': {'abc': 'P n c 2', 'cab': 'P 2 n a', 'bca': 'P b 2 n',
                    'acb': 'P n 2 b', 'bac': 'P c n 2', 'cba': 'P 2 a n'},
        'P m n 21': {'abc': 'P m n 21', 'cab': 'P 21 m n', 'bca': 'P n 21 m',
                     'acb': 'P m 21 n', 'bac': 'P n m 21', 'cba': 'P 21 n m'},
        'P b a 2': {'abc': 'P b a 2', 'cab': 'P 2 c b', 'bca': 'P c 2 a',
                    'acb': 'P c 2 a', 'bac': 'P b a 2', 'cba': 'P 2 c b'},
        'P n a 21': {'abc': 'P n a 21', 'cab': 'P 21 n b', 'bca': 'P c 21 n',
                     'acb': 'P n 21 a', 'bac': 'P b n 21', 'cba': 'P 21 c n'},
        'P n n 2': {'abc': 'P n n 2', 'cab': 'P 2 n n', 'bca': 'P n 2 n',
                    'acb': 'P n 2 n', 'bac': 'P n n 2', 'cba': 'P 2 n n'},
        'C m m 2': {'abc': 'C m m 2', 'cab': 'A 2 m m', 'bca': 'B m 2 m',
                    'acb': 'B m 2 m', 'bac': 'C m m 2', 'cba': 'A 2 m m'},
        'C m c 21': {'abc': 'C m c 21', 'cab': 'A 21 m a', 'bca': 'B b 21 m',
                     'acb': 'B m 21 b', 'bac': 'C c m 21', 'cba': 'A 21 a m'},
        'C c c 2': {'abc': 'C c c 2', 'cab': 'A 2 a a', 'bca': 'B b 2 b',
                    'acb': 'B b 2 b', 'bac': 'C c c 2', 'cba': 'A 2 a a'},
        'A m m 2': {'abc': 'A m m 2', 'cab': 'B 2 m m', 'bca': 'C m 2 m',
                    'acb': 'A m 2 m', 'bac': 'B m m 2', 'cba': 'C 2 m m'},
        'A b m 2': {'abc': 'A b m 2', 'cab': 'B 2 c m', 'bca': 'C m 2 a',
                    'acb': 'A c 2 m', 'bac': 'B m a 2', 'cba': 'C 2 m b'},
        'A m a 2': {'abc': 'A m a 2', 'cab': 'B 2 m b', 'bca': 'C c 2 m',
                    'acb': 'A m 2 a', 'bac': 'B b m 2', 'cba': 'C 2 c m'},
        'A b a 2': {'abc': 'A b a 2', 'cab': 'B 2 c b', 'bca': 'C c 2 a',
                    'acb': 'A c 2 a', 'bac': 'B b a 2', 'cba': 'C 2 c b'},
        'F m m 2': {'abc': 'F m m 2', 'cab': 'F 2 m m', 'bca': 'F m 2 m',
                    'acb': 'F m 2 m', 'bac': 'F m m 2', 'cba': 'F 2 m m'},
        'F d d 2': {'abc': 'F d d 2', 'cab': 'F 2 d d', 'bca': 'F d 2 d',
                    'acb': 'F d 2 d', 'bac': 'F d d 2', 'cba': 'F 2 d d'},
        'I m m 2': {'abc': 'I m m 2', 'cab': 'I 2 m m', 'bca': 'I m 2 m',
                    'acb': 'I m 2 m', 'bac': 'I m m 2', 'cba': 'I 2 m m'},
        'I b a 2': {'abc': 'I b a 2', 'cab': 'I 2 c b', 'bca': 'I c 2 a',
                    'acb': 'I c 2 a', 'bac': 'I b a 2', 'cba': 'I 2 c b'},
        'I m a 2': {'abc': 'I m a 2', 'cab': 'I 2 m b', 'bca': 'I c 2 m',
                    'acb': 'I m 2 a', 'bac': 'I b m 2', 'cba': 'I 2 c m'},
        'P m m m': {'abc': 'P m m m', 'cab': 'P m m m', 'bca': 'P m m m',
                    'acb': 'P m m m', 'bac': 'P m m m', 'cba': 'P m m m'},
        'P n n n': {'abc': 'P n n n', 'cab': 'P n n n', 'bca': 'P n n n',
                    'acb': 'P n n n', 'bac': 'P n n n', 'cba': 'P n n n'},
        'P c c m': {'abc': 'P c c m', 'cab': 'P m a a', 'bca': 'P b m b',
                    'acb': 'P b m b', 'bac': 'P c c m', 'cba': 'P m a a'},
        'P b a n': {'abc': 'P b a n', 'cab': 'P n c b', 'bca': 'P c n a',
                    'acb': 'P c n a', 'bac': 'P b a n', 'cba': 'P n c b'},
        'P m m a': {'abc': 'P m m a', 'cab': 'P b m m', 'bca': 'P m c m',
                    'acb': 'P m a m', 'bac': 'P m m b', 'cba': 'P c m m'},
        'P n n a': {'abc': 'P n n a', 'cab': 'P b n n', 'bca': 'P n c n',
                    'acb': 'P n a n', 'bac': 'P n n b', 'cba': 'P c n n'},
        'P m n a': {'abc': 'P m n a', 'cab': 'P b m n', 'bca': 'P n c m',
                    'acb': 'P m a n', 'bac': 'P n m b', 'cba': 'P c n m'},
        'P c c a': {'abc': 'P c c a', 'cab': 'P b a a', 'bca': 'P b c b',
                    'acb': 'P b a b', 'bac': 'P c c b', 'cba': 'P c a a'},
        'P b a m': {'abc': 'P b a m', 'cab': 'P m c b', 'bca': 'P c m a',
                    'acb': 'P c m a', 'bac': 'P b a m', 'cba': 'P m c b'},
        'P c c n': {'abc': 'P c c n', 'cab': 'P n a a', 'bca': 'P b n b',
                    'acb': 'P b n b', 'bac': 'P c c n', 'cba': 'P n a a'},
        'P b c m': {'abc': 'P b c m', 'cab': 'P m c a', 'bca': 'P b m a',
                    'acb': 'P c m b', 'bac': 'P c a m', 'cba': 'P m a b'},
        'P n n m': {'abc': 'P n n m', 'cab': 'P m n n', 'bca': 'P n m n',
                    'acb': 'P n m n', 'bac': 'P n n m', 'cba': 'P m n n'},
        'P m m n': {'abc': 'P m m n', 'cab': 'P n m m', 'bca': 'P m n m',
                    'acb': 'P m n m', 'bac': 'P m m n', 'cba': 'P n m m'},
        'P b c n': {'abc': 'P b c n', 'cab': 'P n c a', 'bca': 'P b n a',
                    'acb': 'P c n b', 'bac': 'P c a n', 'cba': 'P n a b'},
        'P b c a': {'abc': 'P b c a', 'cab': 'P b c a', 'bca': 'P b c a',
                    'acb': 'P c a b', 'bac': 'P c a b', 'cba': 'P c a b'},
        'P n m a': {'abc': 'P n m a', 'cab': 'P b n m', 'bca': 'P m c n',
                    'acb': 'P n a m', 'bac': 'P m n b', 'cba': 'P c m n'},
        'C m c m': {'abc': 'C m c m', 'cab': 'A m m a', 'bca': 'B b m m',
                    'acb': 'B m m b', 'bac': 'C c m m', 'cba': 'A m a m'},
        'C m c a': {'abc': 'C m c a', 'cab': 'A b m a', 'bca': 'B b c m',
                    'acb': 'B m a b', 'bac': 'C c m b', 'cba': 'A c a m'},
        'C m m m': {'abc': 'C m m m', 'cab': 'A m m m', 'bca': 'B m m m',
                    'acb': 'B m m m', 'bac': 'C m m m', 'cba': 'A m m m'},
        'C c c m': {'abc': 'C c c m', 'cab': 'A m a a', 'bca': 'B b m b',
                    'acb': 'B b m b', 'bac': 'C c c m', 'cba': 'A m a a'},
        'C m m a': {'abc': 'C m m a', 'cab': 'A b m m', 'bca': 'B m c m',
                    'acb': 'B m a m', 'bac': 'C m m b', 'cba': 'A c m m'},
        'C c c a': {'abc': 'C c a a', 'cab': 'A b a a', 'bca': 'B b c b',
                    'acb': 'B b a b', 'bac': 'C c c b', 'cba': 'A c a a'},
        'F m m m': {'abc': 'F m m m', 'cab': 'F m m m', 'bca': 'F m m m',
                    'acb': 'F m m m', 'bac': 'F m m m', 'cba': 'F m m m'},
        'F d d d': {'abc': 'F d d d', 'cab': 'F d d d', 'bca': 'F d d d',
                    'acb': 'F d d d', 'bac': 'F d d d', 'cba': 'F d d d'},
        'I m m m': {'abc': 'I m m m', 'cab': 'I m m m', 'bca': 'I m m m',
                    'acb': 'I m m m', 'bac': 'I m m m', 'cba': 'I m m m'},
        'I b a m': {'abc': 'I b a m', 'cab': 'I m c b', 'bca': 'I c m a',
                    'acb': 'I c m a', 'bac': 'I b a m', 'cba': 'I m c b'},
        'I b c a': {'abc': 'I b c a', 'cab': 'I b c a', 'bca': 'I b c a',
                    'acb': 'I c a b', 'bac': 'I c a b', 'cba': 'I c a b'},
        'I m m a': {'abc': 'I m m a', 'cab': 'I b m m', 'bca': 'I m c m',
                    'acb': 'I m a m', 'bac': 'I m m  b', 'cba': 'I c m m'}}
spg2origins = {}
''' A dictionary of all spacegroups that have 2nd settings; the value is the
1st --> 2nd setting transformation vector as X(2nd) = X(1st)-V, nonstandard
ones are included.
'''
spg2origins = {
        'P n n n':[-.25,-.25,-.25],
        'P b a n':[-.25,-.25,0],'P n c b':[0,-.25,-.25],'P c n a':[-.25,0,-.25],
        'P m m n':[-.25,-.25,0],'P n m m':[0,-.25,-.25],'P m n m':[-.25,0,-.25],
        'C c c a':[0,-.25,-.25],'C c c b':[-.25,0,-.25],'A b a a':[-.25,0,-.25],
        'A c a a':[-.25,-.25,0],'B b c b':[-.25,-.25,0],'B b a b':[0,-.25,-.25],
        'F d d d':[-.125,-.125,-.125],
        'P 4/n':[-.25,-.25,0],'P 42/n':[-.25,-.25,-.25],'I 41/a':[0,-.25,-.125],
        'P 4/n b m':[-.25,-.25,0],'P 4/n n c':[-.25,-.25,-.25],'P 4/n m m':[-.25,-.25,0],'P 4/n c c':[-.25,-.25,0],
        'P 42/n b c':[-.25,-.25,-.25],'P 42/n n m':[-.25,.25,-.25],'P 42/n m c':[-.25,.25,-.25],'P 42/n c m':[-.25,.25,-.25],
        'I 41/a m d':[0,.25,-.125],'I 41/a c d':[0,.25,-.125],
        'p n -3':[-.25,-.25,-.25],'F d -3':[-.125,-.125,-.125],'P n -3 n':[-.25,-.25,-.25],
        'P n -3 m':[-.25,-.25,-.25],'F d -3 m':[-.125,-.125,-.125],'F d -3 c':[-.375,-.375,-.375],
        'p n 3':[-.25,-.25,-.25],'F d 3':[-.125,-.125,-.125],'P n 3 n':[-.25,-.25,-.25],
        'P n 3 m':[-.25,-.25,-.25],'F d 3 m':[-.125,-.125,-.125],'F d - c':[-.375,-.375,-.375]}
spglist = {}
'''A dictionary of space groups as ordered and named in the pre-2002
International Tables Volume A, except that spaces are used following the
GSAS convention to separate the different crystallographic directions.
Note that the symmetry codes here will recognize many non-standard space
group symbols with different settings. They are ordered by Laue group
'''
spglist = {
    'P1': ('P 1', 'P -1'),  # 1-2
    'C1': ('C 1', 'C -1'),
    'P2/m': ('P 2', 'P 21', 'P m', 'P a', 'P c', 'P n',
             'P 2/m', 'P 21/m', 'P 2/c', 'P 2/a', 'P 2/n',
             'P 21/c', 'P 21/a', 'P 21/n'),  # 3-15
    'C2/m': ('C 2', 'C m', 'C c', 'C n',
             'C 2/m', 'C 2/c', 'C 2/n'),
    'A2/m': ('A 2', 'A m', 'A a', 'A n',
             'A 2/m', 'A 2/a', 'A 2/n'),
    'I2/m': ('I 2', 'I m', 'I a', 'I n', 'I c',
             'I 2/m', 'I 2/a', 'I 2/c', 'I 2/n'),
    'Pmmm': ('P 2 2 2', 'P 2 2 21', 'P 21 2 2', 'P 2 21 2',
             'P 21 21 2', 'P 2 21 21', 'P 21 2 21', 'P 21 21 21',
             'P m m 2', 'P 2 m m', 'P m 2 m', 'P m c 21',
             'P 21 m a', 'P b 21 m', 'P m 21 b', 'P c m 21',
             'P 21 a m', 'P c c 2', 'P 2 a a', 'P b 2 b',
             'P m a 2', 'P 2 m b', 'P c 2 m', 'P m 2 a',
             'P b m 2', 'P 2 c m', 'P c a 21', 'P 21 a b',
             'P c 21 b', 'P b 21 a', 'P b c 21', 'P 21 c a',
             'P n c 2', 'P 2 n a', 'P b 2 n', 'P n 2 b',
             'P c n 2', 'P 2 a n', 'P m n 21', 'P 21 m n',
             'P n 21 m', 'P m 21 n', 'P n m 21', 'P 21 n m',
             'P b a 2', 'P 2 c b', 'P c 2 a', 'P n a 21',
             'P 21 n b', 'P c 21 n', 'P n 21 a', 'P b n 21',
             'P 21 c n', 'P n n 2', 'P 2 n n', 'P n 2 n',
             'P m m m', 'P n n n', 'P c c m', 'P m a a', 'P b m b',
             'P b a n', 'P n c b', 'P c n a', 'P m m a', 'P b m m',
             'P m c m', 'P m a m', 'P m m b', 'P c m m', 'P n n a',
             'P b n n', 'P n c n', 'P n a n', 'P n n b', 'P c n n',
             'P m n a', 'P b m n', 'P n c m', 'P m a n', 'P n m b',
             'P c n m', 'P c c a', 'P b a a', 'P b c b', 'P b a b',
             'P c c b', 'P c a a', 'P b a m', 'P m c b', 'P c m a',
             'P c c n', 'P n a a', 'P b n b', 'P b c m', 'P m c a',
             'P b m a', 'P c m b', 'P c a m', 'P m a b', 'P n n m',
             'P m n n', 'P n m n', 'P m m n', 'P n m m', 'P m n m',
             'P b c n', 'P n c a', 'P b n a', 'P c n b', 'P c a n',
             'P n a b', 'P b c a', 'P c a b', 'P n m a', 'P b n m',
             'P m c n', 'P n a m', 'P m n b', 'P c m n'),
    'Cmmm': ('C 2 2 21', 'C 2 2 2', 'C m m 2', 'C m c 21', 'C c m 21',
             'C c c 2', 'C m 2 m', 'C 2 m m', 'C m 2 a', 'C 2 m b',
             'C c 2 m', 'C 2 c m', 'C c 2 a', 'C 2 c b', 'C m c m',
             'C c m m', 'C m c a', 'C c m b', 'C m m m', 'C c c m',
             'C m m a', 'C m m b', 'C c c a', 'C c c b'),
    'Ammm': ('A 21 2 2', 'A 2 2 2', 'A 2 m m', 'A 21 m a', 'A 21 a m',
             'A 2 a a', 'A m 2 m', 'A m m 2', 'A b m 2', 'A c 2 m',
             'A m a 2', 'A m 2 a', 'A b a 2', 'A c 2 a', 'A m m a',
             'A m a m', 'A b m a', 'A c a m', 'A m m m', 'A m a a',
             'A b m m', 'A c m m', 'A c a a', 'A b a a'),
    'Bmmm': ('B 2 21 2', 'B 2 2 2', 'B m 2 m', 'B m 21 b', 'B b 21 m',
             'B b 2 b', 'B m m 2', 'B 2 m m', 'B 2 c m', 'B m a 2',
             'B 2 m b', 'B b m 2', 'B 2 c b', 'B b a 2', 'B b m m',
             'B m m b', 'B b c m', 'B m a b', 'B m m m', 'B b m b',
             'B m a m', 'B m c m', 'B b a b', 'B b c b'),
    'Immm': ('I 2 2 2', 'I 21 21 21', 'I m m 2', 'I m 2 m', 'I 2 m m',
             'I b a 2', 'I 2 c b', 'I c 2 a', 'I m a 2', 'I 2 m b', 'I c 2 m',
             'I m 2 a', 'I b m 2', 'I 2 c m', 'I m m m', 'I b a m', 'I m c b',
             'I c m a', 'I b c a', 'I c a b', 'I m m a', 'I b m m ', 'I m c m',
             'I m a m', 'I m m b', 'I c m m'),
    'Fmmm': ('F 2 2 2', 'F m m m', 'F d d d', 'F m m 2', 'F m 2 m', 'F 2 m m',
             'F d d 2', 'F d 2 d', 'F 2 d d'),
    'P4/mmm': ('P 4', 'P 41', 'P 42', 'P 43', 'P -4', 'P 4/m', 'P 42/m',
               'P 4/n', 'P 42/n', 'P 4 2 2', 'P 4 21 2', 'P 41 2 2',
               'P 41 21 2', 'P 42 2 2', 'P 42 21 2', 'P 43 2 2', 'P 43 21 2',
               'P 4 m m', 'P 4 b m', 'P 42 c m', 'P 42 n m', 'P 4 c c',
               'P 4 n c', 'P 42 m c', 'P 42 b c', 'P -4 2 m', 'P -4 2 c',
               'P -4 21 m', 'P -4 21 c', 'P -4 m 2', 'P -4 c 2', 'P -4 b 2',
               'P -4 n 2', 'P 4/m m m', 'P 4/m c c', 'P 4/n b m', 'P 4/n n c',
               'P 4/m b m', 'P 4/m n c', 'P 4/n m m', 'P 4/n c c',
               'P 42/m m c', 'P 42/m c m', 'P 42/n b c', 'P 42/n n m',
               'P 42/m b c', 'P 42/m n m', 'P 42/n m c', 'P 42/n c m'),
    'I4/mmm': ('I 4', 'I 41', 'I -4', 'I 4/m', 'I 41/a', 'I 4 2 2', 'I 41 2 2',
               'I 4 m m', 'I 4 c m', 'I 41 m d', 'I 41 c d', 'I -4 m 2',
               'I -4 c 2', 'I -4 2 m', 'I -4 2 d', 'I 4/m m m', 'I 4/m c m',
               'I 41/a m d', 'I 41/a c d'),
    'R3-H': ('R 3', 'R -3', 'R 3 2', 'R 3 m', 'R 3 c', 'R -3 m', 'R -3 c'),
    'P6/mmm': ('P 3', 'P 31', 'P 32', 'P -3', 'P 3 1 2', 'P 3 2 1', 'P 31 1 2',
               'P 31 2 1', 'P 32 1 2', 'P 32 2 1', 'P 3 m 1', 'P 3 1 m',
               'P 3 c 1', 'P 3 1 c', 'P -3 1 m', 'P -3 1 c', 'P -3 m 1',
               'P -3 c 1', 'P 6', 'P 61', 'P 65', 'P 62', 'P 64', 'P 63',
               'P -6', 'P 6/m', 'P 63/m', 'P 6 2 2', 'P 61 2 2', 'P 65 2 2',
               'P 62 2 2', 'P 64 2 2', 'P 63 2 2', 'P 6 m m', 'P 6 c c',
               'P 63 c m', 'P 63 m c', 'P -6 m 2', 'P -6 c 2', 'P -6 2 m',
               'P -6 2 c', 'P 6/m m m', 'P 6/m c c', 'P 63/m c m',
               'P 63/m m c'),
    'Pm3m': ('P 2 3', 'P 21 3', 'P m 3', 'P m -3', 'P n 3', 'P n -3', 'P a 3',
             'P a -3', 'P 4 3 2', 'P 42 3 2', 'P 43 3 2', 'P 41 3 2',
             'P -4 3 m', 'P -4 3 n', 'P m 3 m', 'P m -3 m', 'P n 3 n',
             'P n -3 n', 'P m 3 n', 'P m -3 n', 'P n 3 m', 'P n -3 m'),
    'Im3m': ('I 2 3', 'I 21 3', 'I m 3', 'I m -3', 'I a 3', 'I a -3',
             'I 4 3 2', 'I 41 3 2', 'I -4 3 m', 'I -4 3 d', 'I m -3 m',
             'I m 3 m', 'I a 3 d', 'I a -3 d', 'I n 3 n', 'I n -3 n'),
    'Fm3m': ('F 2 3', 'F m 3', 'F m -3', 'F d 3', 'F d -3', 'F 4 3 2',
             'F 41 3 2', 'F -4 3 m', 'F -4 3 c', 'F m 3 m', 'F m -3 m',
             'F m 3 c', 'F m -3 c', 'F d 3 m', 'F d -3 m', 'F d 3 c',
             'F d -3 c')}

sgequiv_2002_orthorhombic = {}
''' A dictionary of orthorhombic space groups that were renamed in the 2002 Volume A,
 along with the pre-2002 name. The e designates a double glide-plane
'''
sgequiv_2002_orthorhombic = {
        'AEM2':'A b m 2','B2EM':'B 2 c m','CM2E':'C m 2 a',
        'AE2M':'A c 2 m','BME2':'B m a 2','C2ME':'C 2 m b',
        'AEA2':'A b a 2','B2EB':'B 2 c b','CC2E':'C c 2 a',
        'AE2A':'A c 2 a','BBE2':'B b a 2','C2CE':'C 2 c b',
        'CMCE':'C m c a','AEMA':'A b m a','BBEM':'B b c m',
        'BMEB':'B m a b','CCME':'C c m b','AEAM':'A c a m',
        'CMME':'C m m a','AEMM':'A b m m','BMEM':'B m c m',
        'CCCE':'C c c a','AEAA':'A b a a','BBEB':'B b c b'}

#'A few non-standard space groups for test use'
nonstandard_sglist = ('P 21 1 1','P 1 21 1','P 1 1 21','R 3 r','R 3 2 h', 
                      'R -3 r', 'R 3 2 r','R 3 m h', 'R 3 m r',
                      'R 3 c r','R -3 c r','R -3 m r',),

#Use the space groups types in this order to list the symbols in the 
#order they are listed in the International Tables, vol. A'''
symtypelist = ('triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 
               'trigonal', 'hexagonal', 'cubic')
