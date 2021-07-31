import typing as t
import subprocess
import importlib
from types import MappingProxyType

# Prevent subprocess.Popen from
# outputting to the console during runtime
try:
    from subprocess import DEVNULL
except ImportError:
    # Support for Python2
    import os

    DEVNULL = open(os.devnull, 'wb')


# ----------------------------------------------
# ----- Validation and Custom Exceptions -------
# ----------------------------------------------

class UninstalledDependencyError(Exception):
    def __init__(self, item: str, msg: str = ''):
        super().__init__(f"{item} not installed ... {msg}")


class MissingModuleError(Exception):
    def __init__(self, item: str, msg: str = ''):
        super().__init__(f"python module(s) - '{item}' cannot be found ... {msg}")


class Test:
    """
    CheckList class that checks whether all the components
    required are installed.
    Adding additional todos is simple:
        Just add the command to check whether something is installed
    """

    def __init__(self, ubuntu_commands: t.Dict):
        """
        Create immutable dictionary so that harmful commands cannot be added
        And the values are also tuples, ensuring no harmful commands can be added
        As an additional layer of security (to prevent shell injection),
        shell option is set to False.
        """
        self.ubuntu_todos = MappingProxyType(ubuntu_commands)
        # Check for Python modules to see whether they exist
        self.python_modules = ()

    def run_cmd(self, cmds: t.Tuple):
        """
        Execute command to check whether a particular
        component was installed.
        Note: This method is secure, since it does not use
        shell=True, which exposes the method to script injection.
        :param cmds: List of words in commands, including flags
        """
        has_pipe: bool = '|' in cmds
        if has_pipe:
            index_of_pipe = cmds.index('|')
            # Process command before pipe
            process_before_pipe = subprocess.Popen(cmds[:index_of_pipe], stdout=subprocess.PIPE)
            # handle command after pipe
            subprocess.check_output(cmds[index_of_pipe + 1:], stdin=process_before_pipe.stdout)
            process_before_pipe.wait()
        else:
            subprocess.check_call(cmds, stdout=DEVNULL, stderr=DEVNULL)

    def not_installed(self, component: str) -> bool:
        """
        Check whether a particular component is installed
        and outputs relevant message to the user
        :param component: The component to validate
        E.g. cuda validation
        :return: wrapper function that returns True if component is
        installed. Otherwise, returns false.
        """
        if component not in self.ubuntu_todos:
            raise KeyError(f"Invalid validation component: {component}. "
                           f"Available components to validate: {self.ubuntu_todos.keys()}")

        installed = True
        try:
            self.run_cmd(self.ubuntu_todos[component])
        except FileNotFoundError:
            installed = False
        return not installed

    def check_dependencies(self) -> None:
        """
        Intentionally written out explicitly so
        that people will know exactly what items are checked.
        The following checks are performed ->

        if not cudnn_installed():
            raise NotInstalledError(CUDA)
        if not cuda_installed():
            raise NotInstalledError(CUDNN)
        if not deepstream_installed():
            raise NotInstalledError(DEEPSTREAM)
        """
        uninstalled_items = [checklist_item for checklist_item, commands in
                             self.ubuntu_todos.items() if self.not_installed(checklist_item)]

        uninstalled_python_items = []
        # writing in for loop since we can't do try except in list comprehensions
        for python_module in self.python_modules:
            try:
                importlib.import_module(python_module)
            except ImportError:
                uninstalled_python_items.append(python_module)

        if len(uninstalled_items) > 0:
            raise UninstalledDependencyError(", ".join(uninstalled_items))

        if len(uninstalled_python_items) > 0:
            raise MissingModuleError(", ".join(uninstalled_python_items))


class Downloader:
    def __init__(self, download_path: str):
        self.download_path = download_path

    def download(self) -> bool:
        """
           Download the target dataset onto the local file path
           Returns:
               True if download is successful. Otherwise, if there is an error,
               or partial completion, return False
        """
        # Step 1. check if download path is valid


class CocoDatasetDownloader(Downloader):
    def __init__(self, local_file_path):
        super().__init__()


    def download(self) -> bool:
        """
        Download the cocodataset onto the local file path
        Returns:
            True if download is successful. Otherwise, if there is an error,
            or partial completion, return False
        """

