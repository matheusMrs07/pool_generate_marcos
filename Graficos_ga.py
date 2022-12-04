import matplotlib.pyplot, os
from mpl_toolkits.mplot3d import Axes3D

def grafico_linha(valor1, valor2):

    matplotlib.pyplot.plot(valor1, valor2)
    matplotlib.pyplot.show()

def grafico_disper(nome_base, l, valor1, valor2, i ,gr, pasta, valor3=None,  legend=None):
    fig = matplotlib.pyplot.figure()

    if valor3!=None:
        ax2 = fig.add_subplot(111, projection='3d')
        a = ax2.scatter(valor1, valor2, valor3, c='b', marker='+', alpha=0.4)
        ax2.set_zlabel(l[2])
    else:
        ax2 = fig.add_subplot(111)
        a = ax2.scatter(valor1, valor2, c='b', marker='+', alpha=0.4)
    ax2.set_title(nome_base)
    ax2.set_xlabel(l[0])
    ax2.set_ylabel(l[1])
    if legend!=None:
        ax2.legend([a], [legend])
    if (os.path.exists(pasta+"/"+str(i)) == False):
         os.system("mkdir -p "+pasta+"/"+str(i))
    fig.savefig(pasta+"/"+str(i)+"/" + nome_base +str(gr)+ ".png")

    return
