import pstats

p = pstats.Stats('EG_components.dat')
p.strip_dirs().sort_stats('cumulative').print_stats(20)
