from events import *
from plotting import *
from multiprocessing import Pool

def problem_2(v_0,N,count,seed = 42):

    np.random.seed(seed)
    
    ensemble = Ensemble(N,0.0005)
    theta = np.random.random(N) * 2 * np.pi
    v = v_0 * np.array([np.cos(theta),np.sin(theta)])

    ensemble.set_velocities(v)

    # change the mass of half of the particles
    mid = N//2
    ensemble.M[mid:] *= 4

    ensemble.simulate(dt = 1,stopper = "equilibrium",stop_val = count)
    
    return ensemble, v

def problem_2_simple(v_0,N,count = 50):

    ensemble, v = problem_2(v_0,N)
    ensemble.v_0 = v
    problem_2_plot(ensemble,ensemble.M[0])

def problem_2_para(v_0,N,count = 50):
    
    pool = Pool()

    results = [pool.apply_async(problem_2, [v_0,N,count,i]) for i in range(8)]
    answers = [results[i].get(timeout = None) for i in range(8)]

    sum_ensemble           = Ensemble(1)
    sum_ensemble.N         = 8*N
    sum_ensemble.M         = np.concatenate([answers[i][0].M for i in range(8)], axis = 0)
    sum_ensemble.v_0       = np.concatenate([answers[i][1] for i in range(8)], axis = 1)
    sum_ensemble.particles = np.concatenate([answers[i][0].particles for i in range(8)], axis = 1)

    problem_2_plot(sum_ensemble,sum_ensemble.M[0])

def problem_2_plot(ensemble,m):

    fig, ax = plt.subplots(ncols = 2, figsize = (20,7))

    v_0     = ensemble.v_0
    v_0_abs = np.sqrt(np.einsum('ij,ij->j',v_0,v_0))
    v2      = ensemble.get_v_square() 
    v_abs   = np.sqrt(v2)

    light = ensemble.M == m
    heavy = ensemble.M == 4*m

    kT_l  = np.average( m/2 * v2[light] )
    kT_h  = np.average( 2 * m * v2[heavy] )
        
    v = np.linspace(0,np.max(v_abs),1000)

    xlim = [np.min(v_abs), np.max(v_abs)]
    
    ax[0].set_title(r"Initial velocity distribution")

    v_0_l = v_0_abs[light]
    v_0_h = v_0_abs[heavy]

    v_0_l[0] *= 2
    v_0_h[0] *= 2
    
    sns.histplot(v_0_l,
                 stat = "density",
                 color = "blue",
                 ax = ax[0],
                 edgecolor = None,
                 label = r"mass $= m$",
                 alpha = 0.7)
    sns.histplot(v_0_h,
                 stat = "density",
                 color = "red",
                 ax = ax[0],
                 edgecolor = None,
                 label = r"mass $= 4m$",
                 alpha = 0.7)
    
    ax[0].grid(ls = "--")
    ax[0].set_xlabel(r"$v$")
    ax[0].set_ylabel(r"Particle density")
    ax[0].set_xlim(xlim)

    ax[0].legend()
    
    plt.tight_layout()

    ax[1].set_title(r"Final velocity distribution")
    
    sns.histplot(v_abs[light],
                 stat = "density",
                 color = "blue",
                 ax = ax[1],
                 edgecolor = None,
                 label = r"mass $=m$",
                 alpha = 0.7)
    ax[1].plot(v,boltzmann_dist(kT_l,m,v),
             label = r"$p(v) = \frac{mv}{kT} \exp{\left(-\frac{m v^2}{2kT}\right)}$",
             color = "blue",
             ls = "--")
    
    sns.histplot(v_abs[heavy],
                 stat = "density",
                 color = "red",
                 ax = ax[1],
                 edgecolor = None,
                 label = r"mass $=4m$",
                 alpha = 0.7)
    ax[1].plot(v,boltzmann_dist(kT_h,4*m,v),
             label = r"$p(v) = \frac{4mv}{kT} \exp{\left(-\frac{4m v^2}{2kT}\right)}$",
             color = "red",
             ls = "--")

    ax[1].grid(ls = "--")

    ax[1].set_xlabel(r"$v$")
    ax[1].set_ylabel(r"Particle density")
    
    plt.legend()
    plt.tight_layout()

    fig.savefig("../fig/distribution_2.pdf")

    v_avg_h = np.average(v_abs[heavy])
    T_avg_h = np.average( 4*m * v2[heavy] / 2 )

    v_avg_l = np.average(v_abs[light])
    T_avg_l = np.average( m/2 * v2[light] )

    table = PrettyTable()

    table.field_names = ["Mass","Average velocity","Average energy"]
    table.add_row(["m", "%.6f"%v_avg_l,"%.6f"%T_avg_l])
    table.add_row(["4m","%.6f"%v_avg_h,"%.6f"%T_avg_h])

    text_file = open("../data/averages.txt","w")
    text_file.write(table.get_string())
    text_file.close()

