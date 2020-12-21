import logging
import functools
import itertools
import re
import copy

import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mathtools.pose import RigidTransform


logger = logging.getLogger(__name__)


# --=( MAIN CLASSES )==--------------------------------------------------------
class Mesh(object):
    def __init__(self, vertices, faces, textures=None, color=None):
        if color is not None:
            texture_size = 2
            textures = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3))
            textures[..., :] = color

        self.vertices = vertices
        self.faces = faces
        self.textures = textures


class Link(object):
    def __init__(self, name, pose=None, mesh=None):
        self.name = name
        self._transform = pose
        self.mesh = mesh

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
        if isinstance(links, list) or isinstance(links, tuple):
            links = {link.name: link for link in links}
        elif not isinstance(links, dict):
            err_str = f"Links has type {type(links)}, but should be dict, list, or tuple"
            raise AssertionError(err_str)

        if isinstance(joints, list) or isinstance(joints, tuple):
            joints = {joint.name: joint for joint in joints}
        elif not isinstance(joints, dict):
            err_str = f"Joints has type {type(joints)}, but should be dict, list, or tuple"
            raise AssertionError(err_str)

        self.links = links
        self.joints = joints

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

    def compute_link_poses(self):
        edges = self.adjacency_matrix
        visited = np.zeros(edges.shape[0], dtype=np.bool)
        for link_name, link in self.links.items():
            link_index = link_name  # FIXME

            stack = [link_index]
            while stack:
                index = stack.pop()
                visited[index] = True
                for neighbor_idx in edges[index, :].nonzero()[0]:
                    neighbor_pose = self.pose  # + joint.pose  # FIXME
                    if visited[neighbor_idx]:
                        if neighbor_pose != self.links[neighbor_idx].pose:
                            raise AssertionError()
                    else:
                        self.links[neighbor_idx].pose = neighbor_pose
                    stack.append(neighbor_idx)

    @property
    def adjacency_matrix(self):
        num_links = len(self.links)
        adjacencies = np.zeros((num_links, num_links), dtype=bool)
        for joint_name, joint in self.joints.items():
            parent_index = joint.parent_name  # FIXME
            child_index = joint.child_name    # FIXME
            adjacencies[parent_index, child_index] = True
        return adjacencies

    def add_joint(self, parent, child, directed=True, in_place=False, transform=None):
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
        joint = Joint(joint_name, 'fixed', parent, child, transform=transform)
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
            if not self.joints[name].joint_type == 'imaginary'
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

    def __add__(self, other):
        # FIXME: this won't detect or avoid collisions
        sum_ = copy.deepcopy(self)
        for joint in other.joints.values():
            cur_joint = sum_.joints.get(joint.name, None)
            if cur_joint is None:
                sum_.joints[joint.name] = joint
            elif cur_joint != joint:
                raise AssertionError(f"{joint.name} inconsistent between self and other")

        return sum_

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
        joint_names = tuple(f"{key}" for key in self.joints.keys())
        return '{' + ', '.join(joint_names) + '}'

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

    @classmethod
    def from_blockassembly(cls, *args, **kwargs):
        return _from_blockassembly(*args, **kwargs)

    def to_blockassembly(cls, *args, **kwargs):
        return _to_blockassembly(*args, **kwargs)


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
        for i, a in enumerate(assemblies):
            f.write(f'ASSEMBLY {i}' + '\n')
            f.write(str(a) + '\n')
            f.write('----------' + '\n')


def render(renderer, assembly, t, R):
    t = t.permute(1, 0)
    R = R.permute(2, 1, 0)

    if R.shape[-1] != t.shape[-1]:
        err_str = f"R shape {R.shape} doesn't match t shape {t.shape}"
        raise AssertionError(err_str)

    num_templates = R.shape[-1]

    # component_poses = tuple(
    #     (np.eye(3), np.zeros(3))
    #     for k in assembly.connected_components.keys()
    # )
    # assembly = assembly.setPose(component_poses, in_place=False)
    init_vertices = torch.stack(
        tuple(link.mesh.vertices for l_name, link in assembly.links.items()),
        dim=0
    )
    faces = torch.stack(
        tuple(link.mesh.faces for l_name, link in assembly.links.items()),
        dim=0
    )
    textures = torch.stack(
        tuple(link.mesh.textures for l_name, link in assembly.links.items()),
        dim=0
    )

    vertices = torch.einsum('nvj,jit->nvit', [init_vertices, R]) + t
    vertices = vertices.permute(-1, 0, 1, 2)

    faces = faces.expand(num_templates, *faces.shape)
    textures = textures.expand(num_templates, *textures.shape)

    rgb_images_obj, depth_images_obj = renderer.render(
        torch.reshape(vertices, (-1, *vertices.shape[2:])),
        torch.reshape(faces, (-1, *faces.shape[2:])),
        torch.reshape(textures, (-1, *textures.shape[2:]))
    )
    rgb_images_scene, depth_images_scene, label_images_scene = render.reduceByDepth(
        torch.reshape(rgb_images_obj, vertices.shape[:2] + rgb_images_obj.shape[1:]),
        torch.reshape(depth_images_obj, vertices.shape[:2] + depth_images_obj.shape[1:]),
        max_depth=renderer.far
    )

    return rgb_images_scene, depth_images_scene, label_images_scene


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


def _from_blockassembly(block_assembly):
    def get_all_stud_coords(block):
        top = block.local_stud_coords
        bottom = top - np.array([[0, 0, block.size_z]])
        return np.vstack((top, bottom))

    def make_block_subassembly(block):
        def makeMesh(block):
            return Mesh(block.local_vertices, block.faces, block.textures)

        positions = get_all_stud_coords(block)  # FIXME: convert to mm
        rotation = Rotation.from_euler('Z', 0, degrees=True)

        links = [Link(block.index, pose=None, mesh=makeMesh(block))] + [
            Link((block.index, i), pose=None)
            for i, position in enumerate(positions)
        ]

        joints = [
            Joint(
                (block.index, (block.index, i)), 'fixed',
                block.index, (block.index, i),
                transform=RigidTransform(position, copy.deepcopy(rotation))
            )
            for i, position in enumerate(positions)
        ]
        block = Assembly(links=links, joints=joints, symmetries=None)
        return block

    def combine_block_subassemblies(blocks, block_assembly):
        def makeJoints(parent, child):
            parent_studs = parent.inGlobalFrame(get_all_stud_coords(parent))
            child_studs = child.inGlobalFrame(get_all_stud_coords(child))

            # array has dims: parent studs, child studs, 3
            stud_distances = parent_studs[:, None, :] - child_studs[None, :, :]
            is_connected_ud = (stud_distances == 0).all(axis=2)
            is_connected_ew = (np.abs(stud_distances) == np.array([[[1, 0, 0]]])).all(axis=2)
            is_connected_ns = (np.abs(stud_distances) == np.array([[[0, 1, 0]]])).all(axis=2)
            is_connected = is_connected_ud | is_connected_ew | is_connected_ns
            stud_edges = np.column_stack(np.nonzero(is_connected))

            joints = {
                ((parent.index, parent_stud_id), (child.index, child_stud_id)): Joint(
                    ((parent.index, parent_stud_id), (child.index, child_stud_id)),
                    'fixed',
                    (parent.index, parent_stud_id),
                    (child.index, child_stud_id),
                    transform=None
                )
                for parent_stud_id, child_stud_id in stud_edges
            }
            joints[((parent.index), (child.index))] = Joint(
                ((parent.index), (child.index)),
                'imaginary', parent.index, child.index,
                transform=getTransform(parent, child)
            )
            return joints

        def getTransform(parent, child):
            t_parent, theta_z_parent = parent.getPose()
            t_child, theta_z_child = child.getPose()
            t_diff = t_parent - t_child

            theta_z_diff = theta_z_parent - theta_z_child
            R_diff = Rotation.from_euler('Z', theta_z_diff, degrees=True)

            transform = RigidTransform(t_diff, R_diff)
            return transform

        links = {}
        for name, block in blocks.items():
            links.update(block.links)

        joints = {}
        for name, block in blocks.items():
            joints.update(block.joints)

        edges = np.column_stack(np.nonzero(block_assembly.symmetrized_connections))
        for parent_index, child_index in edges:
            interblock_joints = makeJoints(
                block_assembly.blocks[parent_index],
                block_assembly.blocks[child_index]
            )
            joints.update(interblock_joints)

        assembly = Assembly(links=links, joints=joints, symmetries=None)
        return assembly

    block_subassemblies = {
        index: make_block_subassembly(block)
        for index, block in block_assembly.blocks.items()
    }

    assembly = combine_block_subassemblies(block_subassemblies, block_assembly)
    return assembly


def _to_blockassembly(block_assembly):
    raise NotImplementedError()


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
