package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	SecondDerivDisp = NewVectorField("force", "", "Force/volume", calcSecondDerivDisp)
	C11             = NewScalarParam("C11", "N/m2", "Stiffness constant C11")
	C12             = NewScalarParam("C12", "N/m2", "Stiffness constant C12")
	C44             = NewScalarParam("C44", "N/m2", "Stiffness constant C44")
	GhostNodes      = false
)

func init() {
	DeclVar("GhostNodes", &GhostNodes, "")
}

func calcSecondDerivDisp(dst *data.Slice) {

	if GhostNodes {
		uMesh := MeshOf(&U)
		u := cuda.SetGhostCells(U.Buffer(), uMesh)
		defer cuda.Recycle(u)

		c1 := C11.MSlice()
		defer c1.Recycle()
		c11 := cuda.SetGhostCellsParameters(c1, C11.Mesh())
		defer c11.Recycle()

		c2 := C12.MSlice()
		defer c2.Recycle()
		c22 := cuda.SetGhostCellsParameters(c2, C12.Mesh())
		defer c22.Recycle()

		c3 := C44.MSlice()
		defer c3.Recycle()
		c33 := cuda.SetGhostCellsParameters(c3, C44.Mesh())
		defer c33.Recycle()

		dstGhost := cuda.Buffer(u.NComp(), u.Size())
		SecondDerivative(dstGhost, u, c11, c22, c33)
		offsetX := 0
		offsetY := 0
		offsetZ := 0
		if !(uMesh.PBC()[X] == 0) {
			offsetX += 1
		}
		if !(uMesh.PBC()[Y] == 0) {
			offsetY += 1
		}
		if !(uMesh.PBC()[Z] == 0) {
			offsetZ += 1
		}
		cuda.Crop(dst, dstGhost, offsetX, offsetY, offsetZ)
		cuda.Recycle(dstGhost)
	} else {
		c1 := C11.MSlice()
		defer c1.Recycle()

		c2 := C12.MSlice()
		defer c2.Recycle()

		c3 := C44.MSlice()
		defer c3.Recycle()
		SecondDerivative(dst, U.Buffer(), c1, c2, c3)
	}
}

func SecondDerivative(dst, u *data.Slice, C1, C2, C3 cuda.MSlice) {
	cuda.SecondDerivative(dst, u, U.Mesh(), C1, C2, C3)
}
