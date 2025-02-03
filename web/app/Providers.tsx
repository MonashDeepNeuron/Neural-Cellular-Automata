"use client";

import React, { type ReactNode } from "react";
import { Provider } from "react-redux";
import store from "../store/store";

interface ProvidersProps {
	children: ReactNode;
}

export default function Providers({ children }: ProvidersProps) {
	return <Provider store={store}>{children}</Provider>;
}
