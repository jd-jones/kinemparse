import logging
import functools
import itertools

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transforms import Rotation


logger = logging.getLogger(__name__)


def RigidMotion(object):
    def __init__(self, translation=None, rotation=None):
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = Rotation.from_matrix(np.eye(3))

        self._t = translation
        self._R = rotation

    def __eq__(self, other):
        translations_equal = self._t == other._t
        rotations_equal = self._R == other._R
        return translations_equal and rotations_equal

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.apply(other)
        if isinstance(other, RigidMotion):
            return self.compose(other)

    def compose(self, other):
        R = self._R * other._R
        t = R.apply(other._t) + self._t
        return RigidMotion(translation=t, rotation=R)

    def inv(self, vector):
        R = self._R.inv()
        t = -self._R.apply(self._t)
        return RigidMotion(translation=t, rotation=R)

    def apply(self, vector):
        return self._t + self._R.apply(vector)


def Link(object):
    def __init__(self, name, pose=None):
        self.name = name
        self._transform = pose

    def __eq__(self, other):
        names_equal = self.name == other.name
        # poses_equal = self.pose == other.pose
        return names_equal

    @property
    def pose(self):
        return self._transform


def Joint(object):
    def __init__(self, name, joint_type, parent_name, child_name, transform=None):
        self.name = name
        self.joint_type = joint_type

        self.parent_name = parent_name
        self.child_name = child_name

        self._transform = transform

    def _params(self):
        params = self.name, self.joint_type, self.parent_name, self.child_name, self.pose
        return params

    def __eq__(self, other):
        return all(p_self == p_other for p_self, p_other in zip(self._params, other._params))

    @property
    def transform(self):
        return self._transform


def Assembly(object):
    def __init__(self, links=[], joints=[], symmetries=None):
        self.links = {link.name: link for link in links}
        self.joints = {joint.name: joint for joint in joints}

        for joint in self.joints:
            child = self.links[joint.child_name]
            parent = self.link[joint.parent_name]
            child._transform = parent._transform * joint._transform

        self.symmetries = symmetries

    @property
    def link_names(self):
        return tuple(self.links.keys())

    @property
    def joint_names(self):
        return tuple(self.joints.keys())

    def __eq__(self, other):
        if self.symmetries is not None:
            raise NotImplementedError()

        if self.link_names != other.link_names:
            return False
        links_equal = all(self.links[name] == other.links[name] for name in self.link_names)

        if self.joint_names != other.joint_names:
            return False
        joints_equal = all(self.joints[name] == other.joints[name] for name in self.joint_names)

        return links_equal and joints_equal

    @classmethod
    def from_xacro(cls, *args, **kwargs):
        return _from_xacro(*args, **kwargs)


def AssemblyAction(Assembly):
    pass


# --=( HELPER FUNCTIONS )==----------------------------------------------------
def _from_xacro(xacro_fn):
    XML_PREFIXES = {"xacro": "http://www.ros.org/wiki/xacro"}
    GLOBAL_PARAMS = {'pi': np.pi}

    def load_macro(fn):
        root = ET.parse(fn).getroot()
        macro_root = root.find('xacro:macro', XML_PREFIXES)
        macro_name = macro_root.get("name")

        return macro_name, macro_root

    def expand_params(attr_str, params):
        for param_name, param_val in params.items():
            attr_str.replace('${param_name}', f"{param_val}")

    def parse_part(macro_root, params=None):
        for element in macro_root.iter():
            for attr_name, attr_val in element.items():
                new_val = expand_params(attr_val, params)
                element.set(attr_name, new_val)

        links = [makeLink(entry, params=params) for entry in macro_root.findall("link")]
        joints = [makeJoint(entry, params=params) for entry in macro_root.findall("joint")]

        return links, joints

    def gen_subassemblies(root):
        entries = root.findall('xacro:include', XML_PREFIXES)
        part_macros = {
            name: functools.partial(parse_part, root)
            for name, root in map(lambda x: x.get('filename'), entries)
        }

        for entry in root.iter():
            for macro_name, macro in part_macros.items():
                if entry.tag.endswith(macro_name):
                    params = GLOBAL_PARAMS.copy()
                    params.update(entry.items)
                    yield macro(params=params)
                    break

    def makeTranslation(xyz):
        xyz = list(map(float, xyz))
        return np.array(xyz)

    def makeRotation(rpy):
        rpy = list(map(float, rpy))
        return Rotation.from_euler('zyx', np.array(rpy))

    def makeLink(entry):
        name = entry.get("name")

        origin = entry.find("visual").find("origin")
        if origin is None:
            pose = None
        else:
            xyz = origin.get("xyz").split(" ")
            rpy = origin.get("rpy").split(" ")
            pose = RigidMotion(
                translation=makeTranslation(xyz),
                rotation=makeRotation(rpy)
            )

        return Link(name, pose=pose)

    def makeJoint(entry):
        name = entry.get("name")
        joint_type = entry.get("type")

        parent_name = entry.find("parent").get("link")
        child_name = entry.find("child").get("link")

        origin = entry.find("origin")
        if origin is None:
            transform = None
        else:
            xyz = origin.get("xyz").split(" ")
            rpy = origin.get("rpy").split(" ")
            transform = RigidMotion(
                translation=makeTranslation(xyz),
                rotation=makeRotation(rpy)
            )

        return Joint(name, joint_type, parent_name, child_name, transform=transform)

    root = ET.parse(xacro_fn).getroot()

    toplevel_links, toplevel_joints = parse_part(root, params=None)
    links, joints = (itertools.chain(*tups) for tups in zip(*gen_subassemblies(root)))
    links = toplevel_links + links
    joints = toplevel_joints + joints
    assembly = Assembly(links=links, joints=joints)

    return assembly
