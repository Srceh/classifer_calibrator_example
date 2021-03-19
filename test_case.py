import numpy
import tqdm
import scipy
import scipy.stats
import matplotlib
import matplotlib.pyplot
from joblib import dump, load
import warnings
from calib import iso, EMP
from dirichlet_python.dirichletcal.calib.tempscaling import TemperatureScaling
from dirichlet_python.dirichletcal.calib.vectorscaling import VectorScaling
from dirichlet_python.dirichletcal.calib.matrixscaling import MatrixScaling
from dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
from sklearn.linear_model import LogisticRegression


# specify the dataset settings (a mixutre of gaussian)
pi = numpy.array([0.5, 0.5])
ita_0 = numpy.array([0.5, 0.5])
ita_1 = numpy.array([0.5, 0.5])
mu_0_0 = numpy.array([4.0])
mu_0_1 = numpy.array([-64.0])
mu_1_0 = numpy.array([-4.0])
mu_1_1 = numpy.array([72.0])
cov_0_0 = numpy.array([960.0])
cov_0_1 = numpy.array([1280.0])
cov_1_0 = numpy.array([980.0])
cov_1_1 = numpy.array([1024.0])

pi3 = numpy.ones(3) / 3
ita_2 = numpy.array([0.5, 0.5])
mu_2_0 = numpy.array([0.0])
mu_2_1 = numpy.array([0.0])
cov_2_0 = numpy.array([8.0])
cov_2_1 = numpy.array([8.0])


def test_prdict_proba(test_case, x):
    p_x_y = numpy.vstack([pi[0] * ita_0[0] * scipy.stats.multivariate_normal.pdf(x=x, mean=mu_0_0, cov=cov_0_0) +
                          pi[1] * ita_0[1] * scipy.stats.multivariate_normal.pdf(x=x, mean=mu_0_1, cov=cov_0_1),
                          pi[1] * ita_1[0] * scipy.stats.multivariate_normal.pdf(x=x, mean=mu_1_0, cov=cov_1_0) +
                          pi[1] * ita_1[1] * scipy.stats.multivariate_normal.pdf(x=x, mean=mu_1_1, cov=cov_1_1)])
    p_x = numpy.sum(p_x_y, axis=0)

    res = numpy.transpose(p_x_y / p_x)

    if test_case == 2:
        res[:, 1] = 1 / (1 + numpy.exp(-0.05*x.ravel()))
        res[:, 0] = 1 - res[:, 1]
    elif test_case == 1:
        # res[:, 1] = numpy.clip((res[:, 1] - 0.5) * 0.5 + 0.5, 0, 1)
        # res[:, 0] = 1 - res[:, 1]
        res[:, 1] = 1 / (1 + numpy.exp(-0.01 * x.ravel()))
        res[:, 0] = 1 - res[:, 1]
    else:
        mdl = load('models/mlp.joblib')
        res = mdl.predict_proba(x)

    return res


def get_binary_data(N=10000):
    y = scipy.stats.multinomial.rvs(p=numpy.hstack([pi[0] * ita_0, pi[1] * ita_1]), size=N, n=1)
    x = [scipy.stats.multivariate_normal.rvs(mean=mu_0_0, cov=cov_0_0, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_0_1, cov=cov_0_1, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_1_0, cov=cov_1_0, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_1_1, cov=cov_1_1, size=N).reshape(-1, 1)]
    x = x[0] * y[:, 0].reshape(-1, 1) + x[1] * y[:, 1].reshape(-1, 1) + \
        x[2] * y[:, 2].reshape(-1, 1) + x[3] * y[:, 3].reshape(-1, 1)
    y = numpy.vstack([y[:, 0] + y[:, 1],
                      y[:, 2] + y[:, 3]]).transpose()
    return x, y


def draw_test_case(test_case, cal_ref):

    N = 10000

    if cal_ref == 'EMP10':
        cal = EMP(10)
    elif cal_ref == 'EMP32':
        cal = EMP(32)
    elif cal_ref == 'ISO_L':
        cal = iso()
    elif cal_ref == 'ISO_S':
        N = 1000
        cal = iso()
    elif cal_ref == 'PLATT':
        cal = LogisticRegression()
    elif cal_ref == 'BETA':
        cal = FullDirichletCalibrator()
    
    x = numpy.load('./data/x' + '_' + str(N) + '.npy')

    y = numpy.load('./data/y' + '_' + str(N) + '.npy')

    s = test_prdict_proba(test_case, x)

    # get predicted scores on an interval
    N_grid = 128
    edge = 160.0
    v_edge = 160.0
    x_mesh = numpy.linspace(-edge, edge, N_grid).reshape(-1, 1)
    s_mesh = test_prdict_proba(test_case, x_mesh)

    # calculate the corresponding densities and probabilities
    p_x_y = numpy.vstack([pi[0] * ita_0[0] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_0_0, cov=cov_0_0) +
                          pi[1] * ita_0[1] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_0_1, cov=cov_0_1),
                          pi[1] * ita_1[0] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_1_0, cov=cov_1_0) +
                          pi[1] * ita_1[1] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_1_1, cov=cov_1_1)])

    p_x = numpy.sum(p_x_y, axis=0)

    p_true = p_x_y / p_x

    if cal_ref == 'EMP10':
        cal.fit(s, y)
    elif cal_ref == 'EMP32':
        cal.fit(s, y)
    elif cal_ref == 'ISO_L':
        cal.fit(s, y)
    elif cal_ref == 'ISO_S':
        cal.fit(s, y)
    elif cal_ref == 'PLATT':
        cal.fit(s, numpy.argmax(y, axis=1))
    elif cal_ref == 'BETA':
        cal.fit(s, numpy.argmax(y, axis=1))

    s_mesh_hat = cal.predict_proba(s_mesh)

    numpy.save('./data/case' + str(test_case) + '_' + cal_ref + '_s_mesh.npy', s_mesh)
    numpy.save('./data/case' + str(test_case) + '_' + cal_ref + '_s_mesh_hat.npy', s_mesh_hat)

    matplotlib.pyplot.figure(dpi=128, figsize=(5, 5))
    matplotlib.pyplot.plot(-x_mesh, s_mesh[:, 0], 'k', linewidth=3.0)
    if cal_ref[0] == 'E':
        matplotlib.pyplot.plot(-x_mesh, s_mesh_hat[:, 0], 'cs', markersize=3)
    else:
        matplotlib.pyplot.plot(-x_mesh, s_mesh_hat[:, 0], 'c', linewidth=3)
    matplotlib.pyplot.plot(-x_mesh, p_true[0], 'k', linewidth=3.0, alpha=0.25)
    matplotlib.pyplot.plot(-x[y[:, 0] == 1], numpy.zeros(numpy.sum(y[:, 0] == 1)) - 2e-2, 'bo', markersize=3,
                           alpha=0.01)
    matplotlib.pyplot.plot(-x[y[:, 0] == 0], numpy.zeros(numpy.sum(y[:, 0] == 0)) - 4e-2, 'ro', markersize=3,
                           alpha=0.01)
    matplotlib.pyplot.xlim([-edge, edge])
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('s')
    matplotlib.pyplot.title('Case' + str(test_case) + '-' + cal_ref + ': prediction')
    matplotlib.pyplot.ylim([-0.1, 1.1])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend(['before calibration', 'after calibration', 'true model'])
    matplotlib.pyplot.savefig('./figures/' + 'Case' + str(test_case) + '_' + cal_ref + '_prediction')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N = int(4096)
        N_sim = int(1e4)

        p_s = numpy.zeros(N_grid - 1)
        p_s_0 = numpy.zeros(N_grid - 1)
        p_s_1 = numpy.zeros(N_grid - 1)

        p_s_hat = numpy.zeros(N_grid - 1)
        p_s_0_hat = numpy.zeros(N_grid - 1)
        p_s_1_hat = numpy.zeros(N_grid - 1)

        for i in tqdm.trange(0, N_sim):
            tmp_y = scipy.stats.multinomial.rvs(p=numpy.hstack([pi[0] * ita_0, pi[1] * ita_1]), size=N, n=1)

            tmp_x = [scipy.stats.multivariate_normal.rvs(mean=mu_0_0, cov=cov_0_0, size=N).reshape(-1, 1),
                     scipy.stats.multivariate_normal.rvs(mean=mu_0_1, cov=cov_0_1, size=N).reshape(-1, 1),
                     scipy.stats.multivariate_normal.rvs(mean=mu_1_0, cov=cov_1_0, size=N).reshape(-1, 1),
                     scipy.stats.multivariate_normal.rvs(mean=mu_1_1, cov=cov_1_1, size=N).reshape(-1, 1)]

            tmp_x = tmp_x[0] * tmp_y[:, 0].reshape(-1, 1) + tmp_x[1] * tmp_y[:, 1].reshape(-1, 1) + \
                    tmp_x[2] * tmp_y[:, 2].reshape(-1, 1) + tmp_x[3] * tmp_y[:, 3].reshape(-1, 1)

            tmp_y = numpy.vstack([tmp_y[:, 0] + tmp_y[:, 1],
                                  tmp_y[:, 2] + tmp_y[:, 3]]).transpose()

            tmp_s = test_prdict_proba(test_case, tmp_x)

            p_s = p_s + numpy.histogram(tmp_s[:, 0], bins=numpy.linspace(0, 1, N_grid))[0]
            p_s_0 = p_s_0 + numpy.histogram(tmp_s[tmp_y[:, 0] == 1, 0], bins=numpy.linspace(0, 1, N_grid))[0]
            p_s_1 = p_s_1 + numpy.histogram(tmp_s[tmp_y[:, 1] == 1, 0], bins=numpy.linspace(0, 1, N_grid))[0]
            
            tmp_s = cal.predict_proba(tmp_s)

            p_s_hat = p_s_hat + numpy.histogram(tmp_s[:, 0], bins=numpy.linspace(0, 1, N_grid))[0]
            p_s_0_hat = p_s_0_hat + numpy.histogram(tmp_s[tmp_y[:, 0] == 1, 0], bins=numpy.linspace(0, 1, N_grid))[0]
            p_s_1_hat = p_s_1_hat + numpy.histogram(tmp_s[tmp_y[:, 1] == 1, 0], bins=numpy.linspace(0, 1, N_grid))[0]

    # calculate related probabilities
    
    p_x_y = numpy.vstack([pi[0] * p_s_0.ravel(),
                          pi[1] * p_s_1.ravel()])

    p_x = numpy.sum(p_x_y, axis=0)

    p_y_hat = p_x_y / p_x
    
    # 

    p_x_y_hat = numpy.vstack([pi[0] * p_s_0_hat.ravel(),
                              pi[1] * p_s_1_hat.ravel()])

    p_x_hat = numpy.sum(p_x_y_hat, axis=0)

    p_y_hat_hat = p_x_y_hat / p_x_hat

    # draw reliability diagram

    numpy.save('./data/case' + str(test_case) + '_' + cal_ref + '_p_y_hat.npy', p_y_hat)
    numpy.save('./data/case' + str(test_case) + '_' + cal_ref + '_p_y_hat_hat.npy', p_y_hat_hat)

    matplotlib.pyplot.figure(dpi=128, figsize=(5, 5))
    matplotlib.pyplot.plot(numpy.linspace(0, 1, N_grid - 1), p_y_hat[0, :], 'ks', linewidth=3)
    matplotlib.pyplot.plot(numpy.linspace(0, 1, N_grid - 1), p_y_hat_hat[0, :], 'cs', markersize=5)
    matplotlib.pyplot.plot(numpy.linspace(0, 1, N_grid - 1), numpy.linspace(0, 1, N_grid - 1), 'k', linewidth=3,
                           alpha=0.5)
    matplotlib.pyplot.xlabel('s')
    matplotlib.pyplot.ylabel('p(y=1 | s)')
    matplotlib.pyplot.title('Case' + str(test_case) + '-' + cal_ref + ': reliability diagram')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend(['before calibration', 'after calibration', 'calibrated reference'])
    matplotlib.pyplot.savefig('./figures/' + 'Case' + str(test_case) + '_' + cal_ref + '_reliability')

    # draw calibration map

    s_list = numpy.linspace(1e-8, 1 - 1e-8, 1024)
    s_hat = cal.predict_proba(numpy.hstack([s_list.reshape(-1, 1), 1 - s_list.reshape(-1, 1)]))

    numpy.save('./data/case' + str(test_case) + '_' + cal_ref + '_s_hat.npy', s_hat)

    matplotlib.pyplot.figure(dpi=128, figsize=(5, 5))
    if cal_ref[0] == 'E':
        matplotlib.pyplot.plot(s_list, s_hat[:, 0], 'cs', linewidth=5.0)
    else:
        matplotlib.pyplot.plot(s_list, s_hat[:, 0], 'c', linewidth=5.0)
    matplotlib.pyplot.xlabel('s')
    matplotlib.pyplot.ylabel('s (after calibration)')
    matplotlib.pyplot.title('Case' + str(test_case) + '-' + cal_ref + ': calibration map')
    matplotlib.pyplot.ylim([-0.08, 1.01])
    matplotlib.pyplot.grid()
    matplotlib.pyplot.savefig('./figures/' + 'Case' + str(test_case) + '_' + cal_ref + '_calibration')


def get_ternary_data(N=10000):
    

    y = scipy.stats.multinomial.rvs(p=numpy.hstack([pi3[0] * ita_0, pi3[1] * ita_1, pi3[2] * ita_2]), size=N, n=1)
    x = [scipy.stats.multivariate_normal.rvs(mean=mu_0_0, cov=cov_0_0, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_0_1, cov=cov_0_1, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_1_0, cov=cov_1_0, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_1_1, cov=cov_1_1, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_2_0, cov=cov_2_0, size=N).reshape(-1, 1),
         scipy.stats.multivariate_normal.rvs(mean=mu_2_1, cov=cov_2_1, size=N).reshape(-1, 1)]

    x = x[0] * y[:, 0].reshape(-1, 1) + x[1] * y[:, 1].reshape(-1, 1) + \
        x[2] * y[:, 2].reshape(-1, 1) + x[3] * y[:, 3].reshape(-1, 1) + \
        x[4] * y[:, 4].reshape(-1, 1) + x[5] * y[:, 5].reshape(-1, 1)

    y = numpy.vstack([y[:, 0] + y[:, 1],
                      y[:, 2] + y[:, 3],
                      y[:, 4] + y[:, 5]]).transpose()

    return x, y


def draw_test_case3(cal_ref):
    
    N = 10000
    
    if cal_ref == 'TS':
        cal = TemperatureScaling()
    elif cal_ref == 'VS':
        cal = VectorScaling()
    elif cal_ref == 'MS':
        cal = MatrixScaling()
    elif cal_ref == 'DIR':
        cal = FullDirichletCalibrator()

    # specify the visualisation settings

    N_grid = 50
    edge = 160.0
    v_edge = 160.0
    x_mesh = numpy.linspace(-edge, edge, N_grid).reshape(-1, 1)
    x_mesh_raw = numpy.linspace(-edge, edge, N_grid).reshape(-1, 1)

    # calculate the corresponding densities and probabilities

    p_x_y = numpy.vstack([pi3[0] * ita_0[0] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_0_0, cov=cov_0_0) +
                          pi3[1] * ita_0[1] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_0_1, cov=cov_0_1),
                          pi3[1] * ita_1[0] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_1_0, cov=cov_1_0) +
                          pi3[1] * ita_1[1] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_1_1, cov=cov_1_1),
                          pi3[2] * ita_2[0] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_2_0, cov=cov_2_0) +
                          pi3[2] * ita_2[1] * scipy.stats.multivariate_normal.pdf(x=x_mesh, mean=mu_2_1, cov=cov_2_1)])

    p_x = numpy.sum(p_x_y, axis=0)

    p_y_g_x = p_x_y / p_x

    p_x = p_x
    p_x_y = p_x_y
    p_x_g_y = p_x_y / pi3.reshape(-1, 1)

    x = numpy.load('./data/x3' + '_' + str(N) + '.npy')

    y = numpy.load('./data/y3' + '_' + str(N) + '.npy')

    mdl = load('models/mlp_3.joblib')

    # get predicted scores on samples
    s = mdl.predict_proba(x)

    # cal = MatrixScaling()
    cal.fit(s, numpy.argmax(y, axis=1))

    # get predicted scores on an interval
    N_grid = 50
    edge = 160.0
    v_edge = 160.0
    x_mesh = numpy.linspace(-edge, edge, N_grid).reshape(-1, 1)
    s_mesh = mdl.predict_proba(x_mesh)
    s_cal_mesh = cal.predict_proba(s_mesh)

    numpy.save('./data/case_' + cal_ref + '_s_mesh.npy', s_mesh)
    numpy.save('./data/case_' + cal_ref + '_s_cal_mesh.npy', s_cal_mesh)

    fig, ax_list = matplotlib.pyplot.subplots(ncols=3, nrows=1,
                                              dpi=128, figsize=(16, 4),
                                              sharex=True)

    for i in range(0, 3):
        ax_list[i].plot(-x_mesh, s_mesh[:, i], 'k', linewidth=3.0)
        ax_list[i].plot(-x_mesh, s_cal_mesh[:, i], 'c', linewidth=3.0)
        ax_list[i].plot(-x_mesh_raw, p_y_g_x[i], 'k', linewidth=3.0, alpha=0.5)
        ax_list[i].plot(-x[y[:, 0] == 1], numpy.zeros(numpy.sum(y[:, 0] == 1)) - 2e-2, 'bo', markersize=3, alpha=0.01)
        ax_list[i].plot(-x[y[:, 0] == 0], numpy.zeros(numpy.sum(y[:, 0] == 0)) - 4e-2, 'ro', markersize=3, alpha=0.01)
        ax_list[i].plot(-x[y[:, 2] == 1], numpy.zeros(numpy.sum(y[:, 2] == 1)) - 6e-2, 'go', markersize=3, alpha=0.01)
        # matplotlib.pyplot.ylim([0, 1])
        ax_list[i].set_xlabel('x')
        ax_list[i].set_ylabel('s')
        ax_list[i].set_title('class ' + str(i + 1) + '-' + cal_ref + ': prediction')
        ax_list[i].set_ylim([-0.1, 1.1])
        ax_list[i].grid()
    ax_list[0].legend(['before calibration', 'after calibration', 'calibrated reference'])
    matplotlib.pyplot.savefig('./figures/' + cal_ref + '_prediction')

    N_grid = 1024
    simp_1_grid = numpy.linspace(0.0, 2.0, N_grid)
    simp_2_grid = numpy.linspace(0.0, numpy.sqrt(3), N_grid)
    simp_1_mesh, simp_2_mesh = numpy.meshgrid(simp_1_grid, simp_2_grid)
    simp_mesh = numpy.hstack([simp_1_mesh.reshape(-1, 1), simp_2_mesh.reshape(-1, 1)])
    s_mesh = numpy.hstack([(simp_mesh[:, 1] / numpy.sqrt(3)).reshape(-1, 1),
                           ((simp_mesh[:, 0] - (simp_mesh[:, 1] / numpy.sqrt(3))) / 2).reshape(-1, 1)])
    s_mesh = numpy.hstack([s_mesh, 1 - numpy.sum(s_mesh, axis=1).reshape(-1, 1)])

    s_hat = cal.predict_proba(s_mesh)

    s_mesh_valid = s_mesh[(numpy.sum(s_mesh, axis=1) <= 1.0) & (numpy.min(s_mesh, axis=1) >= 0.0), :]
    s_hat_valid = s_hat[(numpy.sum(s_mesh, axis=1) <= 1.0) & (numpy.min(s_mesh, axis=1) >= 0.0), :]

    vmax = numpy.max(numpy.abs(s_hat_valid - s_mesh_valid)) * 0.15

    numpy.save('./data/case_' + cal_ref + '_s_hat.npy', s_mesh)
    numpy.save('./data/case_' + cal_ref + '_s_mesh.npy', s_cal_mesh)

    fig, ax_list = matplotlib.pyplot.subplots(ncols=3, nrows=1,
                                              dpi=128, figsize=(16, 4))

    for i in range(0, 3):
        ax_list[i].fill_between(numpy.linspace(0, 1, 8), numpy.ones(8) * numpy.sqrt(3),
                                numpy.linspace(0, 1, 8) * numpy.sqrt(3),
                                facecolor='w', edgecolor='k', zorder=1)
        ax_list[i].fill_between(numpy.linspace(1, 2, 8), numpy.ones(8) * numpy.sqrt(3),
                                numpy.linspace(1, 2, 8) * -numpy.sqrt(3) + 2 * numpy.sqrt(3),
                                facecolor='w', edgecolor='k', zorder=1)
        ax_list[i].fill_between(numpy.linspace(0, 2, 8), numpy.zeros(8) * numpy.sqrt(3), numpy.ones(8) * numpy.sqrt(3),
                                facecolor='w', edgecolor='w', zorder=-1)
        im = ax_list[i].imshow((s_hat[:, i] - s_mesh[:, i]).reshape(N_grid, N_grid), origin='lower', cmap='PuOr_r',
                               extent=[0, 2, 0, numpy.sqrt(3)], vmax=vmax, vmin=-vmax)
        ax_list[i].set_xticks([])
        ax_list[i].set_title('class ' + str(i + 1) + '-' + cal_ref + ': calibration map')
        ax_list[i].set_yticks([])
        ax_list[i].text(x=1.05, y=numpy.sqrt(3)-0.1, s='C1')
        ax_list[i].text(x=0.2, y=-0.1, s='C3')
        ax_list[i].text(x=1.8, y=-0.1, s='C2')


    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, ax=ax_list.ravel().tolist(), orientation="horizontal", shrink=0.75)
    matplotlib.pyplot.savefig('./figures/' + cal_ref + '_calibration')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        N = int(4096)
        N_sim = int(1e4)

        p_s_0 = numpy.zeros((3, N_grid - 1))
        p_s_1 = numpy.zeros((3, N_grid - 1))

        p_s_0_hat = numpy.zeros((3, N_grid - 1))
        p_s_1_hat = numpy.zeros((3, N_grid - 1))

        for j in tqdm.trange(0, N_sim):
            y = scipy.stats.multinomial.rvs(p=numpy.hstack([pi3[0] * ita_0, pi3[1] * ita_1, pi3[2] * ita_2]), size=N, n=1)
            x = [scipy.stats.multivariate_normal.rvs(mean=mu_0_0, cov=cov_0_0, size=N).reshape(-1, 1),
                 scipy.stats.multivariate_normal.rvs(mean=mu_0_1, cov=cov_0_1, size=N).reshape(-1, 1),
                 scipy.stats.multivariate_normal.rvs(mean=mu_1_0, cov=cov_1_0, size=N).reshape(-1, 1),
                 scipy.stats.multivariate_normal.rvs(mean=mu_1_1, cov=cov_1_1, size=N).reshape(-1, 1),
                 scipy.stats.multivariate_normal.rvs(mean=mu_2_0, cov=cov_2_0, size=N).reshape(-1, 1),
                 scipy.stats.multivariate_normal.rvs(mean=mu_2_1, cov=cov_2_1, size=N).reshape(-1, 1)]

            x = x[0] * y[:, 0].reshape(-1, 1) + x[1] * y[:, 1].reshape(-1, 1) + \
                x[2] * y[:, 2].reshape(-1, 1) + x[3] * y[:, 3].reshape(-1, 1) + \
                x[4] * y[:, 4].reshape(-1, 1) + x[5] * y[:, 5].reshape(-1, 1)

            y = numpy.vstack([y[:, 0] + y[:, 1],
                              y[:, 2] + y[:, 3],
                              y[:, 4] + y[:, 5]]).transpose()

            tmp_s = mdl.predict_proba(x)

            tmp_ss = cal.predict_proba(tmp_s)

            for i in range(0, 3):
                p_s_0[i, :] = p_s_0[i, :] + numpy.histogram(tmp_s[y[:, i] == 1, i], bins=numpy.linspace(0, 1, N_grid))[
                    0]
                p_s_1[i, :] = p_s_1[i, :] + numpy.histogram(tmp_s[y[:, i] != 1, i], bins=numpy.linspace(0, 1, N_grid))[
                    0]

                p_s_0_hat[i, :] = p_s_0_hat[i, :] + \
                                  numpy.histogram(tmp_ss[y[:, i] == 1, i], bins=numpy.linspace(0, 1, N_grid))[0]
                p_s_1_hat[i, :] = p_s_1_hat[i, :] + \
                                  numpy.histogram(tmp_ss[y[:, i] != 1, i], bins=numpy.linspace(0, 1, N_grid))[0]

    fig, ax_list = matplotlib.pyplot.subplots(ncols=3, nrows=1,
                                              dpi=128, figsize=(16, 4),
                                              sharex=True)

    for i in range(0, 3):
        p_x_y_hat = numpy.vstack([pi3[i] * p_s_0[i, :].ravel(),
                                  (1 - pi3[i]) * p_s_1[i, :].ravel()])

        p_x_hat = numpy.sum(p_x_y_hat, axis=0)

        p_y_hat = p_x_y_hat / p_x_hat

        p_x_y_hat = numpy.vstack([pi3[i] * p_s_0_hat[i, :].ravel(),
                                  (1 - pi3[i]) * p_s_1_hat[i, :].ravel()])

        p_x_hat = numpy.sum(p_x_y_hat, axis=0)

        p_y_hat_hat = p_x_y_hat / p_x_hat

        numpy.save('./data/case_' + cal_ref + '_p_y_hat' + '_' +  str(i) + '.npy', p_y_hat)
        numpy.save('./data/case_' + cal_ref + '_p_y_hat_hat' + '_' +  str(i) + '.npy', p_y_hat_hat)

        ax_list[i].plot(numpy.linspace(0, 1, N_grid - 1), p_y_hat[0, :], 'ks', linewidth=3)
        ax_list[i].plot(numpy.linspace(0, 1, N_grid - 1), p_y_hat_hat[0, :], 'cs', markersize=5)
        ax_list[i].plot(numpy.linspace(0, 1, N_grid - 1), numpy.linspace(0, 1, N_grid - 1), 'k', linewidth=3,
                        alpha=0.5)
        ax_list[i].set_xlabel('s')
        ax_list[i].set_ylabel('p(y=1 | s)')
        ax_list[i].set_title('class ' + str(i + 1) + '-' + cal_ref + ': reliability diagram')
        ax_list[i].grid()
    ax_list[0].legend(['before calibration', 'after calibration', 'calibrated reference'])
    matplotlib.pyplot.savefig('./figures/' + cal_ref + '_reliability')