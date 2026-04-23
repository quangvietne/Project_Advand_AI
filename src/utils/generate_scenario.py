"""Generate a simple SUMO intersection scenario for testing."""
import os


def generate_simple_intersection(output_dir: str = "data/scenarios/hn_sample") -> None:
    """Create a basic 4-way intersection with Vietnamese traffic flows.
    
    Based on team member's SUMO setup code.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Node file (.nod.xml) ---
    with open(f"{output_dir}/nodes.nod.xml", "w", encoding="utf-8") as f:
        f.write("""<nodes>
    <node id="c" x="0" y="0" type="traffic_light"/>
    <node id="n" x="0" y="100"/>
    <node id="s" x="0" y="-100"/>
    <node id="e" x="100" y="0"/>
    <node id="w" x="-100" y="0"/>
</nodes>
""")

    # --- 2. Edge file (.edg.xml) ---
    with open(f"{output_dir}/edges.edg.xml", "w", encoding="utf-8") as f:
        f.write("""<edges>
    <edge id="n2c" from="n" to="c" numLanes="2" speed="13.9"/>
    <edge id="s2c" from="s" to="c" numLanes="2" speed="13.9"/>
    <edge id="e2c" from="e" to="c" numLanes="2" speed="13.9"/>
    <edge id="w2c" from="w" to="c" numLanes="2" speed="13.9"/>

    <edge id="c2n" from="c" to="n" numLanes="2" speed="13.9"/>
    <edge id="c2s" from="c" to="s" numLanes="2" speed="13.9"/>
    <edge id="c2e" from="c" to="e" numLanes="2" speed="13.9"/>
    <edge id="c2w" from="c" to="w" numLanes="2" speed="13.9"/>
</edges>
""")

    # --- 3. Route file (.rou.xml) with VN vehicle types ---
    with open(f"{output_dir}/routes.rou.xml", "w", encoding="utf-8") as f:
        f.write("""<routes>
    <!-- Vietnamese vehicle types: emphasize motorcycles -->
    <vType id="motorcycle" accel="3.0" decel="5.0" sigma="0.3" length="2.0" minGap="1.0" maxSpeed="16.67" vClass="motorcycle"/>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="13.89" vClass="passenger"/>
    <vType id="bus" accel="1.2" decel="3.5" sigma="0.5" length="12.0" minGap="3.0" maxSpeed="11.11" vClass="bus"/>
    <vType id="truck" accel="1.5" decel="3.0" sigma="0.5" length="7.5" minGap="2.5" maxSpeed="11.11" vClass="truck"/>

    <!-- Routes: through movements -->
    <route id="n2s" edges="n2c c2s"/>
    <route id="s2n" edges="s2c c2n"/>
    <route id="e2w" edges="e2c c2w"/>
    <route id="w2e" edges="w2c c2e"/>

    <!-- Traffic flows: EW (truc chinh) = dung 2x NS (truc phu) -->
    <!-- Ti le luu luong EW:NS = 2:1 (chinh xac) -->
    <!-- NS: moto 0.15 + car 0.075 + bus 0.025 = 0.25/s/huong -->
    <!-- EW: moto 0.30 + car 0.15  + truck 0.05 = 0.50/s/huong = 2x NS -->

    <!-- Truc PHU Bac-Nam -->
    <flow id="flow_n2s_moto" type="motorcycle" route="n2s" begin="0" end="3660" probability="0.15"/>
    <flow id="flow_n2s_car" type="car" route="n2s" begin="0" end="3660" probability="0.075"/>
    <flow id="flow_n2s_bus" type="bus" route="n2s" begin="0" end="3660" probability="0.025"/>

    <flow id="flow_s2n_moto" type="motorcycle" route="s2n" begin="0" end="3660" probability="0.15"/>
    <flow id="flow_s2n_car" type="car" route="s2n" begin="0" end="3660" probability="0.075"/>

    <!-- Truc CHINH Dong-Tay (2x NS) -->
    <flow id="flow_e2w_moto" type="motorcycle" route="e2w" begin="0" end="3660" probability="0.30"/>
    <flow id="flow_e2w_car" type="car" route="e2w" begin="0" end="3660" probability="0.15"/>
    <flow id="flow_e2w_truck" type="truck" route="e2w" begin="0" end="3660" probability="0.05"/>

    <flow id="flow_w2e_moto" type="motorcycle" route="w2e" begin="0" end="3660" probability="0.30"/>
    <flow id="flow_w2e_car" type="car" route="w2e" begin="0" end="3660" probability="0.15"/>
</routes>
""")

    # --- 4. SUMO config (.sumocfg) ---
    with open(f"{output_dir}/config.sumocfg", "w", encoding="utf-8") as f:
        f.write("""<configuration>
    <input>
        <net-file value="intersection.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3660"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
    </processing>
</configuration>
""")

    print(f"✓ Created scenario files in {output_dir}/")
    print(f"  Next: run 'netconvert --node-files={output_dir}/nodes.nod.xml --edge-files={output_dir}/edges.edg.xml --output-file={output_dir}/intersection.net.xml'")


if __name__ == "__main__":
    generate_simple_intersection()
