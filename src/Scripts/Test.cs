using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WeArt.Components;

public class Test : MonoBehaviour
{
    [Header("References")]
    [Tooltip("Human Virtual Sphere")]
    public VirtualSphere h_sphere;
    [Tooltip("Robot Virtual Sphere")]
    public VirtualSphere r_sphere;
    
    [Header("Validation Settings")]
    [Tooltip("1.0 for normalized comparison.")]
    public float springStiffness = 1.0f;
    [Tooltip("File name for the CSV report.")]
    public string fileName = "GraspValidationReport.csv";

    // Internal state
    private bool _isRecording = false;
    private float _initialRadiusH;
    private float _initialRadiusR;
    private float _scalingFactor; // k_sc = r_h / r_r

    // Data storage for CSV
    private StringBuilder _csvContent = new StringBuilder();

    void Start()
    {
        // Initialize CSV header
        _csvContent.AppendLine("Time,HumanRadius,RobotRadius,ScalingFactor,HumanEnergy,RobotEnergy,EnergyErrorPercent");
    }

    void Update()
    {
        // Controls to Start/Stop Recording
        if (Input.GetKeyDown(KeyCode.R))
        {
            StartRecording();
        }
        if (Input.GetKeyDown(KeyCode.S))
        {
            StopRecording();
        }

        if (_isRecording)
        {
            PerformValidationStep();
        }
    }

  
    void StartRecording()
    {
        if (h_sphere == null || r_sphere == null)
        {
            Debug.LogError("Validation Error: Please assign both Human and Robot Virtual Spheres in the Inspector.");
            return;
        }

        _initialRadiusH = h_sphere.Radius;
        _initialRadiusR = r_sphere.Radius;

        if (_initialRadiusH > 0)
            _scalingFactor = _initialRadiusR / _initialRadiusH;
        else
            _scalingFactor = 1.0f;

        _csvContent.Clear();
        _csvContent.AppendLine("Time,HumanRadius,RobotRadius,ScalingFactor,HumanEnergy,RobotEnergy,EnergyErrorPercent");
        _isRecording = true;
        Debug.Log($"<color=green>VALIDATION STARTED.</color> Baseline H: {_initialRadiusH:F4}, Baseline R: {_initialRadiusR:F4}, Scaling Factor: {_scalingFactor:F4}");
    }

    void StopRecording()
    {
        _isRecording = false;
        
        string directory;
        directory = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop);
        

        string filePath = Path.Combine(directory, fileName);
        
        try 
        {
            File.WriteAllText(filePath, _csvContent.ToString());
            Debug.Log($"<color=red>VALIDATION STOPPED.</color> Report saved to: {filePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to save file: {e.Message}");
        }
    }

    void PerformValidationStep()
    {
        //  Get current radii
        float currentRadiusH = h_sphere.Radius;
        float currentRadiusR = r_sphere.Radius;

        // Calculate Deformation 
        float deformationH = Mathf.Max(0, _initialRadiusH - currentRadiusH);
        float deformationR = Mathf.Max(0, _initialRadiusR - currentRadiusR);

        // Calculate Elastic Energy
        // E = 0.5 * k * (delta_x)^2
        float energyH = 0.5f * springStiffness * (deformationH * deformationH);
        float energyR = 0.5f * springStiffness * (deformationR * deformationR);

        // Normalize Robot Energy for comparison
        float expectedEnergyR = energyH * (_scalingFactor * _scalingFactor);
        
        // Avoid division by zero
        float errorPercent = 0f;
        if (expectedEnergyR > 1e-6f)
        {
            errorPercent = Mathf.Abs(energyR - expectedEnergyR) / expectedEnergyR * 100.0f;
        }

        // Log Data
        string line = $"{Time.time},{currentRadiusH},{currentRadiusR},{_scalingFactor},{energyH},{energyR},{errorPercent}";
        _csvContent.AppendLine(line);
    }

    // Visual Debugging
    void OnGUI()
    {
        if (_isRecording)
        {
            GUI.Label(new Rect(10, 10, 300, 20), $"Recording Validation... (Press S to Stop)");
            GUI.Label(new Rect(10, 30, 300, 20), $"Current Energy Error: {GetCurrentError():F2}%");
        }
        else
        {
            GUI.Label(new Rect(10, 10, 300, 20), "Press R to Start Validation Recording");
        }
    }

    float GetCurrentError() // for display only
    {
        float dH = Mathf.Max(0, _initialRadiusH - h_sphere.Radius);
        float dR = Mathf.Max(0, _initialRadiusR - r_sphere.Radius);
        float eH = 0.5f * (dH * dH);
        float eR = 0.5f * (dR * dR);
        float expected = eH * (_scalingFactor * _scalingFactor);
        return expected > 1e-5f ? Mathf.Abs(eR - expected) / expected * 100f : 0f;
    }
}