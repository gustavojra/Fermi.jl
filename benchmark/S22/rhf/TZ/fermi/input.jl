using Fermi

# Basic options trying to be uniform

Fermi.set_num_threads()

# Cold run
@set printstyle none
@energy rhf

@set {
    basis cc-pvtz
    scf_e_conv 1e-8
    scf_max_rms 1e-8
    printstyle file
    output output.dat
}

times = Float64[]

for i in 1:22

    # Read in xyz file
    molstring = read("../../xyz/S22-$(i)-dimer.xyz", String)

    # Remove first two lines from xyz
    molstring = String(split(molstring, "\n",limit=3)[end])
    Fermi.Options.set("molstring", molstring)

    # Run energy compt
    t = @elapsed begin
        @energy rhf
    end
    GC.gc()
    # Print out time
    output("@@@ Total Run Time: {}", t)
end



    

    
