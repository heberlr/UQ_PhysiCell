#!/usr/bin/env nextflow

ModelFile = "SampleModel.ini"
KeyModel = "physicell_model"
PythonVersion = "python3"

// Generating XML_Files
process CreateXML {
    cpus 1
    
    input:
    val pythonscript
    
    script:
    """
    $pythonscript
    # main.py
    """
}

// Read .json file with the XML_Files tuple
// Channel
//     .fromPath('./XMLs_Folder/*.xml')

// params.greeting  = 'Hello world!'
// greeting_ch = Channel.of(params.greeting)

// include { SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'

workflow {
    // Channel.of('main.py') | CreateXML
    CreateXML('main.py')
}
