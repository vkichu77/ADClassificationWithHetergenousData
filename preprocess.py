import os
from nipype import Workflow, Node, MapNode, IdentityInterface, Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.interfaces.fsl import BET, FLIRT, MCFLIRT, FNIRT, ApplyWarp, Smooth

def create_workflow(name, data_dir, output_dir):
    # Workflow setup
    wf = Workflow(name=name)
    wf.base_dir = output_dir

    # DataGrabber node to fetch data
    datagrabber = Node(DataGrabber(infields=['subject_id'], outfields=['func', 'struct']),
                       name='datagrabber')
    datagrabber.inputs.base_directory = data_dir
    datagrabber.inputs.template = '*'
    datagrabber.inputs.field_template = {'func': '%s/fmri/*.nii', 'struct': '%s/mri/*.nii'}
    datagrabber.inputs.sort_filelist = True

    # Nodes for MRI processing
    skullstrip = Node(BET(frac=0.5, robust=True), name='skullstrip')
    normalize = Node(FLIRT(reference='/path/to/MNI152_T1_2mm_brain.nii.gz'), name='normalize')
    smooth = Node(Smooth(fwhm=8), name='smooth')

    # Nodes for fMRI processing
    motion_correct = Node(MCFLIRT(mean_vol=True, save_plots=True), name='motion_correct')
    f_normalize = Node(FNIRT(), name='f_normalize')
    f_smooth = Node(Smooth(fwhm=6), name='f_smooth')

    # Connect nodes
    wf.connect([
        (datagrabber, skullstrip, [('struct', 'in_file')]),
        (skullstrip, normalize, [('out_file', 'in_file')]),
        (normalize, smooth, [('out_file', 'in_file')]),
        (datagrabber, motion_correct, [('func', 'in_file')]),
        (motion_correct, f_normalize, [('out_file', 'in_file')]),
        (f_normalize, f_smooth, [('out_file', 'in_file')])
    ])

    # DataSink to collect outputs
    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = os.path.join(output_dir, 'output')
    wf.connect([
        (smooth, datasink, [('smoothed_file', 'mri.@smooth')]),
        (f_smooth, datasink, [('smoothed_file', 'fmri.@smooth')])
    ])

    return wf

# Usage
data_dir = '/path/to/ADNI'
output_dir = '/path/to/output'
workflow = create_workflow('adni_preprocess', data_dir, output_dir)
workflow.run('MultiProc', plugin_args={'n_procs': 4})
