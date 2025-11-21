using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public static class Kinematics
{
    public enum JointType { HingeX, HingeY, HingeZ, HingeXY, Ball }

    public static int GetDoF(Transform[] joints, JointType[] types)
    {
        int dof = 0;
        for (int i=0; i < joints.Length; i++)
        {
            if (joints[i] == null || i >= types.Length) continue;

            var type = types[i];
            if (type == JointType.HingeXY) dof += 2;
            else if (type == JointType.Ball) dof += 3;
            else dof += 1;
        }
        return dof;
    }

    public static Matrix<double> GetJacobian(Transform[] joints, Transform[] refPoints, Transform root, JointType[] types)
    {
        int n = GetDoF(joints, types);
        var J = DenseMatrix.Create(refPoints.Length * 3, n, 0.0);

        int col = 0;
        for (int i=0; i < joints.Length; i++)
        {
            if (joints[i] == null || i >= types.Length) continue;

            Transform joint = joints[i];
            JointType type = types[i];

            for (int k=0; k < refPoints.Length; k++)
            {
                if (refPoints[k] == null) continue;
                Transform effector = refPoints[k];
                if (!effector.IsChildOf(joint)) continue;

                Vector3 r = effector.position - joint.position;
                int row = k*3;

                // x axis
                if (type == JointType.HingeX || type == JointType.HingeXY || type == JointType.Ball)
                {
                    Vector3 axis = Vector3.Cross(joint.right, r);
                    J[row, col] = axis.x;
                    J[row + 1, col] = axis.y;
                    J[row + 2, col] = axis.z;
                }

                // y axis
                if (type == JointType.HingeY || type == JointType.HingeXY || type == JointType.Ball)
                {
                    int c = col + ((type == JointType.HingeXY || type == JointType.Ball) ? 1 : 0);
                    Vector3 axis = Vector3.Cross(joint.up, r);
                    J[row, c] = axis.x;
                    J[row + 1, c] = axis.y;
                    J[row + 2, c] = axis.z;
                }

                // z axis
                if (type == JointType.HingeZ || type == JointType.Ball)
                {
                    int c = col + (type == JointType.Ball ? 2 : 0);
                    Vector3 axis = Vector3.Cross(joint.forward, r);
                    J[row, c] = axis.x;
                    J[row + 1, c] = axis.y;
                    J[row + 2, c] = axis.z;
                }
            }

            if (type == JointType.HingeXY) col += 2;
            else if (type == JointType.Ball) col += 3;
            else col += 1;
        }
        return J;
    }

    public static Vector<double> GetJointAngles(Transform[] joints, JointType[] types)
    {
        int n = GetDoF(joints, types);
        var q = Vector<double>.Build.Dense(n);
        int idx = 0;
        for (int i=0; i < joints.Length; i++)
        {
            if (joints[i] == null || i >= types.Length) continue;
            Vector3 e = joints[i].localEulerAngles;
            var type = types[i];

            float x = (e.x > 180 ? e.x - 360 : e.x) * Mathf.Deg2Rad;
            float y = (e.y > 180 ? e.y - 360 : e.y) * Mathf.Deg2Rad;
            float z = (e.z > 180 ? e.z - 360 : e.z) * Mathf.Deg2Rad;

            if (type == JointType.HingeX || type == JointType.HingeXY || type == JointType.Ball) q[idx++] = x;
            if (type == JointType.HingeY || type == JointType.HingeXY || type == JointType.Ball) q[idx++] = y;
            if (type == JointType.HingeZ || type == JointType.Ball) q[idx++] = z;
        }
        return q;
    }
}
