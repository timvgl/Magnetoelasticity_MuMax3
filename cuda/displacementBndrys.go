package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func SetGhostCells(u *data.Slice, mesh *data.Mesh) *data.Slice {
	if mesh.PBC() == [3]int{0, 0, 0} {
		dst := Buffer(u.NComp(), u.Size())
		data.Copy(dst, u)
		return dst
	}
	util.AssertMsg(u.NComp() == 3, "input data needs to have three components")
	pbc := mesh.PBC()
	size := [3]int{0, 0, 0}
	if pbc[X] == 0 {
		size[X] = u.Size()[X]
	} else {
		size[X] = u.Size()[X] + 2
	}
	if pbc[Y] == 0 {
		size[Y] = u.Size()[Y]
	} else {
		size[Y] = u.Size()[Y] + 2
	}
	if pbc[Z] == 0 {
		size[Z] = u.Size()[Z]
	} else {
		size[Z] = u.Size()[Z] + 2
	}
	dst := Buffer(u.NComp(), size)
	cfg := make3DConf(size)
	k_createExtended3DField_async(dst.DevPtr(0), dst.DevPtr(1), dst.DevPtr(2), u.DevPtr(0), u.DevPtr(1), u.DevPtr(2), u.Size()[X], u.Size()[Y], u.Size()[Z], mesh.PBC_code(), cfg)
	return dst
}

func SetGhostCellsParameters(par MSlice, mesh *data.Mesh) MSlice {
	if par.arr == nil {
		return par
	}
	pbc := mesh.PBC()
	size := [3]int{0, 0, 0}
	if pbc[X] == 0 {
		size[X] = par.Size()[X]
	} else {
		size[X] = par.Size()[X] + 2
	}
	if pbc[Y] == 0 {
		size[Y] = par.Size()[Y]
	} else {
		size[Y] = par.Size()[Y] + 2
	}
	if pbc[Z] == 0 {
		size[Z] = par.Size()[Z]
	} else {
		size[Z] = par.Size()[Z] + 2
	}
	dst := Buffer(par.arr.NComp(), size)
	cfg := make3DConf(size)
	k_extendParameterField_async(dst.DevPtr(0), par.DevPtr(0), par.Size()[X], par.Size()[Y], par.Size()[Z], mesh.PBC_code(), cfg)
	return ToMSlice(dst)
}
