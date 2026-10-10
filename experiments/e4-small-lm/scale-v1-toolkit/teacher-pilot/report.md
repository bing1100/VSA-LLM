# C3 teacher ontology — pilot (decision 65, phase 1b)

400 entries (829 synsets) read by the teacher (`claude-haiku-4-5-20251001, claude-opus-5-5`, prompt `c3-teacher-v1`).

- spend: **$2.41** = $0.0060 per entry, $0.0029 per synset (34 calls, 0 failed; median $0.071 per call)
- time: wall 132 s = 0.33 s per entry (parallel calls); median call 15.8 s; 0.65 call-seconds per synset
- parse: 829 synsets answered, 0 not; 2710 proposed edges, **97.3% mapped** to atomics (ancestor 284, exact 690, lemma 2, lemma_ancestor 2, lexname 829, pos 829); unmapped: no_ancestor 61, self 3, unknown_lemma 10; pos-only fallback frames: 0

## Agreement with the WordNet entry frames

| entries | n | teacher edges / entry | WordNet edges / entry | precision | recall | macro P | macro R | w/o lexname+pos: edges T / W | P | R | filler P | filler R |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| all | 400 | 5.06 | 5.20 | 0.776 | 0.754 | 0.828 | 0.816 | 2.34 / 2.47 | 0.593 | 0.560 | 0.603 | 0.578 |
| heldout | 200 | 6.05 | 6.28 | 0.748 | 0.722 | 0.805 | 0.795 | 2.91 / 3.10 | 0.558 | 0.523 | 0.569 | 0.538 |
| heldout:0 | 40 | 6.78 | 6.85 | 0.793 | 0.785 | 0.862 | 0.841 | 3.25 / 3.42 | 0.662 | 0.628 | 0.672 | 0.636 |
| heldout:1-2 | 40 | 4.72 | 5.05 | 0.767 | 0.718 | 0.818 | 0.786 | 2.15 / 2.45 | 0.558 | 0.490 | 0.558 | 0.500 |
| heldout:10-29 | 40 | 7.47 | 7.92 | 0.742 | 0.700 | 0.807 | 0.794 | 3.75 / 4.10 | 0.553 | 0.506 | 0.561 | 0.522 |
| heldout:3-9 | 40 | 6.03 | 6.45 | 0.730 | 0.682 | 0.760 | 0.737 | 2.92 / 3.35 | 0.538 | 0.470 | 0.547 | 0.481 |
| heldout:30+ | 40 | 5.28 | 5.10 | 0.701 | 0.725 | 0.776 | 0.817 | 2.48 / 2.20 | 0.455 | 0.511 | 0.485 | 0.547 |
| trained | 200 | 4.06 | 4.12 | 0.817 | 0.804 | 0.852 | 0.837 | 1.76 / 1.84 | 0.649 | 0.622 | 0.660 | 0.645 |
| trained:0 | 40 | 3.88 | 3.85 | 0.806 | 0.812 | 0.825 | 0.822 | 1.68 / 1.70 | 0.642 | 0.632 | 0.657 | 0.677 |
| trained:1-9 | 40 | 3.52 | 3.67 | 0.879 | 0.844 | 0.884 | 0.856 | 1.45 / 1.57 | 0.759 | 0.698 | 0.754 | 0.694 |
| trained:10-99 | 40 | 3.58 | 3.60 | 0.839 | 0.833 | 0.855 | 0.843 | 1.50 / 1.52 | 0.683 | 0.672 | 0.695 | 0.695 |
| trained:100-999 | 40 | 4.92 | 5.03 | 0.766 | 0.751 | 0.827 | 0.818 | 2.27 / 2.42 | 0.560 | 0.526 | 0.591 | 0.565 |
| trained:1000+ | 40 | 4.40 | 4.47 | 0.812 | 0.799 | 0.868 | 0.845 | 1.93 / 1.98 | 0.649 | 0.633 | 0.645 | 0.636 |

## Per relation (all sampled entries)

| relation | teacher | WordNet | hits | precision | recall |
|---|---:|---:|---:|---:|---:|
| hypernym | 576 | 570 | 338 | 0.587 | 0.593 |
| instance_hypernym | 54 | 50 | 38 | 0.704 | 0.760 |
| part_meronym | 25 | 51 | 14 | 0.560 | 0.275 |
| member_meronym | 16 | 30 | 10 | 0.625 | 0.333 |
| substance_meronym | 3 | 1 | 0 | 0.000 | 0.000 |
| part_holonym | 45 | 55 | 33 | 0.733 | 0.600 |
| member_holonym | 32 | 42 | 25 | 0.781 | 0.595 |
| substance_holonym | 5 | 5 | 2 | 0.400 | 0.400 |
| attribute | 8 | 4 | 2 | 0.250 | 0.500 |
| similar_to | 61 | 83 | 50 | 0.820 | 0.602 |
| topic_domain | 62 | 37 | 14 | 0.226 | 0.378 |
| entailment | 0 | 12 | 0 | — | 0.000 |
| cause | 4 | 4 | 2 | 0.500 | 0.500 |
| antonym | 44 | 45 | 26 | 0.591 | 0.578 |
| lexname | 627 | 630 | 555 | 0.885 | 0.881 |
| pos | 461 | 461 | 460 | 0.998 | 0.998 |

## Examples (✓ = edge in both frames)

**piece of writing** (heldout, band 3-9, 1 senses)

- WordNet: lexname: noun.communication ✓; pos: n ✓; hypernym: written_communication.n.01 ✓; part_meronym: conclusion.n.08
- teacher: lexname: noun.communication ✓; pos: n ✓; hypernym: written_communication.n.01 ✓

**transparent** (heldout, band 3-9, 4 senses)

- WordNet: lexname: adj.all ✓; pos: s ✓; similar_to: clear.a.04; similar_to: thin.a.01 ✓; similar_to: straight.a.06; similar_to: obvious.a.01
- teacher: lexname: adj.all ✓; pos: s ✓; similar_to: thin.a.01 ✓; similar_to: clear.a.01

**developed** (heldout, band 10-29, 3 senses)

- WordNet: lexname: adj.all ✓; pos: a ✓; pos: s ✓
- teacher: lexname: adj.all ✓; pos: a ✓; pos: s ✓

**experienced** (heldout, band 30+, 1 senses)

- WordNet: lexname: adj.all ✓; pos: a ✓; antonym: inexperienced.a.01 ✓
- teacher: lexname: adj.all ✓; pos: a ✓; attribute: experience.n.03; antonym: inexperienced.a.01 ✓

**get to** (heldout, band 30+, 3 senses)

- WordNet: lexname: verb.emotion ✓; pos: v ✓; lexname: verb.change ✓; hypernym: get_down.v.07; lexname: verb.motion; hypernym: achieve.v.01
- teacher: lexname: verb.emotion ✓; pos: v ✓; lexname: verb.change ✓; hypernym: begin.v.03; lexname: verb.social; hypernym: succeed.v.01

**acarpellous** (trained, band 0, 1 senses)

- WordNet: lexname: adj.all ✓; pos: a ✓; topic_domain: botany.n.02
- teacher: lexname: adj.all ✓; pos: a ✓

**family upupidae** (trained, band 1-9, 1 senses)

- WordNet: lexname: noun.animal ✓; pos: n ✓; hypernym: bird_family.n.01 ✓; member_meronym: coraciiform_bird.n.01; member_meronym: bird_genus.n.01 ✓; member_holonym: coraciiformes.n.01 ✓
- teacher: lexname: noun.animal ✓; pos: n ✓; hypernym: bird_family.n.01 ✓; member_meronym: bird_genus.n.01 ✓; member_holonym: coraciiformes.n.01 ✓

**hand job** (trained, band 1-9, 1 senses)

- WordNet: lexname: noun.act ✓; pos: n ✓; hypernym: sexual_activity.n.01 ✓
- teacher: lexname: noun.act ✓; pos: n ✓; hypernym: sexual_activity.n.01 ✓

**deleterious** (trained, band 1000+, 1 senses)

- WordNet: lexname: adj.all ✓; pos: s ✓; similar_to: harmful.a.01 ✓
- teacher: lexname: adj.all ✓; pos: s ✓; similar_to: harmful.a.01 ✓

**interval** (trained, band 1000+, 4 senses)

- WordNet: lexname: noun.cognition; pos: n ✓; hypernym: set.n.02; lexname: noun.attribute ✓; hypernym: distance.n.01 ✓; lexname: noun.communication; hypernym: musical_notation.n.01 ✓; lexname: noun.time ✓; hypernym: measure.n.02
- teacher: lexname: noun.group; pos: n ✓; hypernym: set.n.01; topic_domain: mathematics.n.01; lexname: noun.attribute ✓; hypernym: distance.n.01 ✓; hypernym: musical_notation.n.01 ✓; topic_domain: music.n.01; lexname: noun.time ✓; hypernym: time.n.05

