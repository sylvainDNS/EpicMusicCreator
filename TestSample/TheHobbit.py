from pysynth_b import *
song = (
    ('a3', 3),
    ('c', 8),
    ('d', 8),
    ('e', 16),
    ('d', 16),
    ('c', 8),
    ('b3', 8),
    ('a3', 3),
    ('r', 8),
    ('e3', 8),
    ('g3', 8),
    ('a3', 8),
    ('b3', 8),
    ('b3', 4),
    ('r', 8),
    ('c', 8),
    ('d', 8),
    ('c', 16),
    ('b3', 16),
    ('a3', 4),
    ('a3', 8),
    ('r', 8),
    ('b3', 8),
    ('c', 6),
    ('b3', 16),
    ('e', 3),
    ('r', 8),
    ('b3', 8),
    ('c', 6),
    ('g3', 16),
    ('a3', 3),
    ('r', 8)
)

make_wav(song, fn="TheHobbit.wav", bpm=48)