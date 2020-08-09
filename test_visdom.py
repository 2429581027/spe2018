from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import argparse
import numpy as np
import math
import os.path
import getpass
import time
from sys import platform as _platform
from six.moves import urllib


def testUpdate():
    try:
        vis = Visdom(port=DEFAULT_PORT, env=env)
        vis.close(None, env=env)

        # 3d scatterplot with custom labels and ranges
        Y = np.random.rand(100)
        sca = vis.scatter(
            X=np.random.rand(100, 3),
            Y=(Y + 1.5).astype(int),
            opts=dict(
                legend=['Men', 'Women'],
                markersize=1,
                xtickmin=0,
                xtickmax=2,
                xlabel='Arbitrary',
                xtickvals=[0, 0.75, 1.6, 2],
                ytickmin=0,
                ytickmax=2,
                ytickstep=0.5,
                ztickmin=0,
                ztickmax=1,
                ztickstep=0.5,
            ),
            name='new_trace',
            #update='new'
        )

        # import time
        # time.sleep(2)

        assert vis.win_exists(sca), 'Created window marked as not existing'

        Y = np.random.rand(10)
        vis.scatter(
            X=np.random.rand(10, 3),
            Y=(Y + 1.5).astype(int),
            win=sca,
            name='new_trace',
            #name='new_trace',
            #update='remove',
            #update='new',
            #update='append',
            #update='replace'
        )

    except BaseException as e:
        print(
            "The visdom experienced an exception while running: {}\n"
            "The demo displays up-to-date functionality with the GitHub version, "
            "which may not yet be pushed to pip. Please upgrade using "
            "`pip install -e .` or `easy_install .`\n"
            "If this does not resolve the problem, please open an issue on "
            "our GitHub.".format(repr(e))
        )
def testPlot2():
    try:
        vis = Visdom(port=DEFAULT_PORT, env=env)
        vis.close(None, env=env)

        #str_update = 'append'
        str_update = 'replace'

        win = vis.line(
                X=np.column_stack([np.arange(0, 1) for i in range(10)]), 
                Y=np.column_stack([np.arange(0, 1) for i in range(10)]),
                win="test"
        )

        vis.line(
        X=np.arange(1, 38),
        Y=np.random.randn(37),
        win=win,
        name='6',
        update=str_update,
        )

        vis.line(
        X=np.arange(1, 38),
        Y=np.random.randn(37),
        win=win,
        name='11',
        update=str_update,
        )


    except BaseException as e:
        print(
            "The visdom experienced an exception while running: {}\n"
            "The demo displays up-to-date functionality with the GitHub version, "
            "which may not yet be pushed to pip. Please upgrade using "
            "`pip install -e .` or `easy_install .`\n"
            "If this does not resolve the problem, please open an issue on "
            "our GitHub.".format(repr(e))
        )

if __name__ == '__main__':
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    env = 'main'    
    #testUpdate()
    testPlot2()