using UnityEngine;
using SEB;
using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace WeArt.Components
{
    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class VirtualSphere : MonoBehaviour
    {
        public Vector3 LocalCenter { get; private set; } = Vector3.zero;
        public float Radius { get; private set; } = 0.05f;
        
        [Range(0.001f, 0.1f)] 
        public float pointSize = 0.01f;

        [Header("Debug")]
        public bool showGizmos = true;
        
        [Header("Configuration")]
        public Transform[] referencePoints;

        private Transform _palm;
        private List<GameObject> _debugPointObjects = new List<GameObject>();

        void Awake()
        {
            _palm = transform.parent;
        }

        void Update()
        {
            if (_palm == null)
            {
                _palm = transform.parent;
                if (_palm == null) return;
            }

            UpdateSphere(_palm);

            transform.localPosition = LocalCenter;
            float diameter = Radius * 2f;
            transform.localScale = Vector3.one * diameter;
        }
        
        void OnDestroy()
        {
            foreach (var dot in _debugPointObjects)
            {
                if (dot != null) Destroy(dot);
            }
        }

        public void UpdateSphere(Transform palm)
        {
            if (palm == null) return;
            _palm = palm; 

            var localPoints = new List<Vector3>();
            foreach (var p in referencePoints) 
            { 
                if (p != null) localPoints.Add(palm.InverseTransformPoint(p.position)); 
            }
            
            if (localPoints.Count < 2) return;

            var set = new ArrayPointSet(3, localPoints.Count); 
            for (int i = 0; i < localPoints.Count; i++)
            {
                set.Set(i, 0, localPoints[i].x);
                set.Set(i, 1, localPoints[i].y);
                set.Set(i, 2, localPoints[i].z);
            }
            var mb = new Miniball(set);

            LocalCenter = new Vector3((float)mb.Center[0], (float)mb.Center[1], (float)mb.Center[2]);
            Radius = (float)mb.Radius;
        }

        public Matrix<double> ComputeMatrixA(Transform palm)
        {
            var localPoints = new List<Vector3>();
            foreach(var p in referencePoints) 
            { 
                if(p != null) localPoints.Add(palm.InverseTransformPoint(p.position)); 
            }

            int n = localPoints.Count;
            if (n == 0) return DenseMatrix.Create(0, 7, 0.0);

            Matrix<double> A = DenseMatrix.Create(3 * n, 7, 0.0);

            for (int i = 0; i < n; i++)
            {
                Vector3 r = localPoints[i] - LocalCenter;
                int row = 3 * i;
                
                // translation (identity)
                A[row, 0] = 1; 
                A[row + 1, 1] = 1; 
                A[row + 2, 2] = 1;
                
                // rotation (skew-symmetric cross product)
                A[row, 3] = 0;    A[row, 4] = -r.z;  A[row, 5] = r.y;
                A[row + 1, 3] = r.z;  A[row + 1, 4] = 0;     A[row + 1, 5] = -r.x;
                A[row + 2, 3] = -r.y; A[row + 2, 4] = r.x;   A[row + 2, 5] = 0;
                
                // radial expansion
                A[row, 6] = r.x; 
                A[row + 1, 6] = r.y; 
                A[row + 2, 6] = r.z;
            }
            return A;
        }

        void OnDrawGizmos()
        {
            if (!showGizmos || referencePoints == null || _palm == null) return;
            if (!Application.isPlaying) UpdateSphere(_palm);

            Vector3 worldCenter = _palm.TransformPoint(LocalCenter);
            float worldRadius = Radius * _palm.lossyScale.x; 

            Gizmos.color = new Color(1, 1, 0, 0.2f);
            Gizmos.DrawSphere(worldCenter, worldRadius);
            Gizmos.DrawWireSphere(worldCenter, worldRadius);
        }
    }
}

        