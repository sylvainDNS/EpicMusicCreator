from mido import MidiFile


mid = MidiFile('./input/test.mid')

song = []
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    trackList = []
    for msg in track:
        note = msg.bytes()
        if(len(note) < 4):
            trackList.append(note)
    song.append(trackList)

print(song[0])
http: // stackabuse.com/tensorflow-neural-network-tutorial/
