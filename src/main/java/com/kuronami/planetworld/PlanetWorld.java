package com.kuronami.planetworld;

import com.kuronami.isekaiapi.api.Isekai;
import com.mojang.logging.LogUtils;
import net.neoforged.bus.api.IEventBus;
import net.neoforged.fml.common.Mod;
import org.slf4j.Logger;

@Mod(PlanetWorld.MODID)
public final class PlanetWorld {
    public static final String MODID = "planet_world";
    public static final String VERSION = "1.0.0";
    public static final Logger LOGGER = LogUtils.getLogger();

    public PlanetWorld(IEventBus modBus) {
        LOGGER.info("Planet World v{} loading", VERSION);
        // Smoke-test the Isekai API facade is reachable at compile time.
        LOGGER.info("Planet World: Isekai API facade ready (query={}, remap={})",
                Isekai.query().getClass().getSimpleName(),
                Isekai.remap().getClass().getSimpleName());
    }
}
