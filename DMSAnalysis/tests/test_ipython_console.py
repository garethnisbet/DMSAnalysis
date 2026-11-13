"""The slider's IPython button: an in-process console sharing the live state."""

import os
import unittest

from .gui_harness import slider_on

CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'configs', 'example_913232.json')

try:
    import qtconsole, ipykernel  # noqa: F401
    HAVE_QTCONSOLE = True
except ImportError:
    HAVE_QTCONSOLE = False


@unittest.skipUnless(HAVE_QTCONSOLE, 'qtconsole/ipykernel not installed')
class TestIPythonConsole(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sl = slider_on(CFG)
        cls.win = cls.sl.win

    @classmethod
    def tearDownClass(cls):
        cls.win._close_console()

    def _shell(self):
        return self.win._console._km.kernel.shell

    def test_namespace_is_the_live_state(self):
        self.win._on_open_console()
        sh = self._shell()
        self.assertIs(sh.user_ns['win'], self.win)
        self.assertIs(sh.user_ns['ts'], self.sl.ts)
        self.assertIs(sh.user_ns['dms'], self.win._dms)
        self.assertIsInstance(sh.user_ns['dms'], self.sl.ts.dmsfit_ico_hkl)
        # slider module globals come along too
        self.assertIn('make_overlay_dms', sh.user_ns)

    def test_code_runs_against_the_engines(self):
        self.win._on_open_console()
        sh = self._shell()
        res = sh.run_cell('_n = len(dms.reflist); _a = win.ig[0]')
        self.assertTrue(res.success, res.error_in_exec)
        self.assertGreater(sh.user_ns['_n'], 0)
        self.assertEqual(sh.user_ns['_a'], self.win.ig[0])

    def test_reopen_reuses_and_refreshes(self):
        self.win._on_open_console()
        con = self.win._console
        sh = self._shell()
        sh.user_ns['dms'] = None
        self.win._on_open_console()
        self.assertIs(self.win._console, con)
        self.assertIs(sh.user_ns['dms'], self.win._dms)
        sh.user_ns['dms'] = None
        sh.run_cell('dms_refresh()')
        self.assertIs(sh.user_ns['dms'], self.win._dms)


if __name__ == '__main__':
    unittest.main()
