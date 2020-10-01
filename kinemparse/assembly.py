import logging
import functools
import itertools
import re
import copy

import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from mathtools.pose import RigidTransform


logger = logging.getLogger(__name__)


# --=( MAIN CLASSES )==--------------------------------------------------------
class Link(object):
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

    def __repr__(self):
        return f"{self.name}: {self._transform}"


class Joint(object):
    def __init__(self, name, joint_type, parent_name, child_name, transform=None):
        self.name = name
        self.joint_type = joint_type

        self.parent_name = parent_name
        self.child_name = child_name

        self._transform = transform

    @property
    def _params(self):
        param_dict = {
            'name': self.name,
            'joint_type': self.joint_type,
            'parent_name': self.parent_name,
            'child_name': self.child_name,
            'transform': self.transform
        }
        return param_dict

    def __eq__(self, other):
        return self._params == other._params

    @property
    def transform(self):
        return self._transform

    def __repr__(self):
        return f"{self.parent_name} -> {self.child_name}: {self._transform}"


class Assembly(object):
    WARN_TRANSFORM = False

    def __init__(self, links=[], joints=[], symmetries=None):
        self.links = {link.name: link for link in links}
        self.joints = {joint.name: joint for joint in joints}

        for joint_name, joint in self.joints.items():
            if joint.joint_type == 'floating':
                logger.info(
                    "Ignoring floating joint: "
                    f"({joint.parent_name} -> {joint.child_name})"
                )
                continue
            child = self.links[joint.child_name]
            parent = self.links[joint.parent_name]
            if joint._transform is None:
                warn_str = (
                    f"For ({joint.parent_name} -> {joint.child_name}), "
                    f"joint transform is {joint._transform}"
                )
                if Assembly.WARN_TRANSFORM:
                    logger.warning(warn_str)
                continue
            if parent._transform is None:
                warn_str = (
                    f"For joint ({joint.parent_name} -> {joint.child_name}), "
                    f"parent pose is {parent._transform}"
                )
                if Assembly.WARN_TRANSFORM:
                    logger.warning(warn_str)
                continue
            child._transform = parent._transform * joint._transform

        self.symmetries = symmetries

    def add_joint(self, parent, child, directed=True, in_place=False):
        if not in_place:
            a_copy = copy.deepcopy(self)
            a_copy = a_copy.add_joint(parent, child, directed=directed, in_place=True)
            return a_copy

        if not directed:
            a = self.add_joint(parent, child, directed=True, in_place=in_place)
            a = a.add_joint(child, parent, directed=True, in_place=in_place)
            return a

        if parent not in self.links:
            self.links[parent] = Link(parent, pose=None)

        if child not in self.links:
            self.links[child] = Link(child, pose=None)

        joint_name = f"{parent}_{child}_joint"
        joint = Joint(joint_name, 'fixed', parent, child, transform=None)
        self.joints[joint_name] = joint

        return self

    def remove_joint(self, parent, child, directed=True, in_place=False):
        if not in_place:
            a_copy = copy.deepcopy(self)
            a_copy = a_copy.remove_joint(parent, child, directed=directed, in_place=True)
            return a_copy

        if not directed:
            a = self.remove_joint(parent, child, directed=True, in_place=in_place)
            a = a.remove_joint(child, parent, directed=True, in_place=in_place)
            return a

        joint_name = f"{parent}_{child}_joint"
        del self.joints[joint_name]

        link_names = tuple(self.links.keys())
        for link_name in link_names:
            for joint in self.joints.values():
                if joint.child_name == link_name:
                    break
                if joint.parent_name == link_name:
                    break
            else:
                del self.links[link_name]

        return self

    @property
    def link_names(self):
        return tuple(self.links.keys())

    @property
    def joint_names(self):
        return tuple(self.joints.keys())

    def getJoints(self, directed=True):
        joints = self.joints.values()
        if not directed:
            joints = remove_sym(joints)
        return joints

    def __le__(self, other):
        if self.symmetries is not None:
            raise NotImplementedError()

        if not set(self.link_names) <= set(other.link_names):
            return False

        links_predicate = all(
            self.links[name] == other.links[name]
            for name in self.link_names
        )

        if not set(self.joint_names) <= set(other.joint_names):
            return False

        joints_predicate = all(
            self.joints[name] == other.joints[name]
            for name in self.joint_names
        )

        return links_predicate and joints_predicate

    def __eq__(self, other):
        return (self <= other) and (other <= self)

    def __lt__(self, other):
        return (self <= other) and not (self == other)

    def __ge__(self, other):
        return other <= self

    def __gt__(self, other):
        return other < self

    def __sub__(self, other):
        if self < other:
            difference = other - self
            if difference:
                difference.sign = -1
            return difference

        if not other <= self:
            raise AssertionError()

        difference = copy.deepcopy(self)
        for joint in other.joints.values():
            difference = difference.remove_joint(
                joint.parent_name, joint.child_name,
                directed=True, in_place=False
            )

        difference = AssemblyAction(
            links=difference.links.values(),
            joints=difference.joints.values(),
            symmetries=difference.symmetries,
            sign=1
        )

        return difference

    def __repr__(self):
        return '{' + ', '.join(self.joints.keys()) + '}'

    def __str__(self):
        link_strs = tuple(link.name for link in self.links.values())
        link_str = f"LINKS: {', '.join(link_strs)}"

        joint_strs = tuple(
            f"({j.parent_name}--{j.child_name})"
            for j in self.getJoints(directed=False)
        )
        joint_str = f"JOINTS: {', '.join(joint_strs)}"

        return link_str + "\n" + joint_str

    def __bool__(self):
        links_exist = bool(self.links)
        joints_exist = bool(self.joints)

        if joints_exist != links_exist:
            raise AssertionError()

        return joints_exist

    @classmethod
    def from_xacro(cls, *args, **kwargs):
        return _from_xacro(*args, **kwargs)


class AssemblyAction(Assembly):
    def __init__(self, sign=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self:
            sign = 0
        self.sign = sign

    def __leq__(self, other):
        signs_eq = self.sign == other.sign
        structure_leq = super().__leq__(self, other)
        return signs_eq and structure_leq

    def __repr__(self):
        return f"{self.action_type} {super().__repr__()}"

    def __str__(self):
        return f"ACTION: {self.action_type}\n{super().__str__()}"

    @property
    def action_type(self):
        if not self.sign:
            return 'NULL'
        if self.sign == 1:
            return 'CONNECT'
        if self.sign == -1:
            return 'DISCONNECT'


# --=( ASSEMBLY FUNCTIONS )==--------------------------------------------------
def writeAssemblies(fn, assemblies):
    with open(fn, 'wt') as f:
        for a in assemblies:
            f.write(str(a) + '\n')
            f.write('----------' + '\n')


# --=( HELPER FUNCTIONS )==----------------------------------------------------
def remove_sym(joints):
    new_joints = []
    for joint in joints:
        if not has_sym(joint, new_joints):
            new_joints.append(joint)
    return new_joints


def has_sym(joint, joints):
    is_sym = (
        (j.parent_name == joint.child_name) and (j.child_name == joint.parent_name)
        for j in joints
    )
    return any(is_sym)


def _from_xacro(xacro_fn, params={}):
    XML_PREFIXES = {"xacro": "http://www.ros.org/wiki/xacro"}
    GLOBAL_PARAMS = {'pi': np.pi}

    def load_macros(fn):
        root = ET.parse(fn).getroot()
        macro_data = tuple(
            (macro_root.get("name"), macro_root)
            for macro_root in root.findall('xacro:macro', XML_PREFIXES)
        )
        return macro_data

    def replace_strings(string, replacements):
        for old, new in replacements.items():
            string = string.replace(old, f"{new}")
        return string

    def expand_params(attr_str, params):
        matches = re.findall(r'\$\{[^\}]+\}', attr_str)
        for match in matches:
            expr_str = match.strip("${").strip("}")
            expr_val = eval(expr_str, params)
            attr_str = attr_str.replace(match, f"{expr_val}")
        return attr_str

    def parse_part(macro_root, params={}):
        macro_root = copy.deepcopy(macro_root)
        for element in macro_root.iter():
            for attr_name, attr_val in element.items():
                new_val = expand_params(attr_val, params)
                element.set(attr_name, new_val)

        links = [makeLink(entry) for entry in macro_root.findall("link")]
        joints = [makeJoint(entry) for entry in macro_root.findall("joint")]

        return links, joints

    def gen_subassemblies(root, params={}):
        entries = root.findall('xacro:include', XML_PREFIXES)
        part_macros = {
            macro_name: functools.partial(parse_part, macro_root)
            for file_name in map(lambda x: replace_strings(x.get('filename'), params), entries)
            for macro_name, macro_root in load_macros(file_name)
        }

        for entry in root.iter():
            def strToFloat(string):
                try:
                    return float(string)
                except ValueError:
                    return string
            entry_params = {name: strToFloat(value) for name, value in entry.items()}
            for macro_name, macro in part_macros.items():
                if entry.tag.endswith(macro_name):
                    params = GLOBAL_PARAMS.copy()
                    params.update(entry_params)
                    result = macro(params=params)
                    yield result
                    break

    def makeTranslation(xyz):
        xyz = list(map(float, xyz))
        return np.array(xyz)

    def makeRotation(rpy):
        rpy = list(map(float, rpy))
        return Rotation.from_euler('xyz', np.array(rpy))
        # return Rotation.from_euler('zyx', np.array(rpy))

    def makeLink(entry):
        name = entry.get("name")

        if entry.find("visual") is None or entry.find("visual").find("origin") is None:
            pose = None
        else:
            origin = entry.find("visual").find("origin")
            xyz = origin.get("xyz").split()
            rpy = origin.get("rpy").split()
            pose = RigidTransform(
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
            xyz = origin.get("xyz").split()
            rpy = origin.get("rpy").split()
            transform = RigidTransform(
                translation=makeTranslation(xyz),
                rotation=makeRotation(rpy)
            )

        return Joint(name, joint_type, parent_name, child_name, transform=transform)

    params.update(GLOBAL_PARAMS)

    root = ET.parse(xacro_fn).getroot()

    toplevel_links, toplevel_joints = parse_part(root, params=params)
    links, joints = (
        list(itertools.chain(*tups))
        for tups in zip(*gen_subassemblies(root, params=params))
    )
    links = toplevel_links + links
    joints = toplevel_joints + joints
    assembly = Assembly(links=links, joints=joints)

    return assembly
